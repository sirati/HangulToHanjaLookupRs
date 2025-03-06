#![feature(random)]

use anyhow::{Result, anyhow};
use csv::{ReaderBuilder, WriterBuilder};
use dialoguer::Select;
use dialoguer::console::{Key, Term};
use dialoguer::term_access::TokioChannelConsoleAccess;
use futures::stream::{FuturesUnordered, StreamExt};
use http_cache_reqwest::{CACacheManager, Cache, CacheMode, HttpCache, HttpCacheOptions};
use rand::random;
use reqwest::Response;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use scraper::{Html, Selector};
use std::cmp::max;
use std::collections::VecDeque;
use std::fmt::Display;
use std::ops::Mul;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::{collections::HashMap, fs::File, io::BufReader, path::Path, time::Duration};
use tokio::select;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::time::{Instant as TokioInstant, sleep_until};

const INPUT_FILE: &str = "/home/sirati/Vocab Unit 1 audio only.txt";
const OUTPUT_FILE: &str = "/home/sirati/Vocab Unit 1 audio only hanja.tsv";
const COLUMN_HANGUL: usize = 1; // second column
const COLUMN_ENGLISH: usize = 2; // third column

macro_rules! aprintln {
    ($dst:expr, $($arg:tt)*) => {
        _println_async(format!($dst, $($arg)*))
    };
    ($dst:expr) => {
        _println_async($dst.to_string())
    };
}

/// Represents a fetch request for a word pair.
#[derive(Debug)]
pub struct FetchRequest {
    index: usize,
    hangul: String,
    eng_word: String,
    url: String,
}

/// Represents the outcome of an HTTP fetch.
#[derive(Debug)]
pub struct FetchResult {
    index: usize,
    hangul: String,
    eng_word: String,
    response: Option<Response>,
}

/// A pending job tracked by the scheduler. Note that backoff state is stored here.
struct PendingJob {
    req: FetchRequest,
}

/// A helper async function that runs one fetch job using `fetch_with_retry`.
/// (Note that here the built-in retry is simplified; the scheduler itself
/// is also responsible for requeueing failed jobs.)
async fn run_fetch_job(
    client: ClientWithMiddleware,
    req: FetchRequest,
) -> (ClientWithMiddleware, FetchRequest, Option<Response>) {
    // Note: the internal retry here is minimal.
    let result = match client.get(&req.url).send().await {
        Ok(resp) if resp.status() == 200 => Some(resp),
        Ok(resp) => {
            eprintln!("Got status {} for URL {}", resp.status(), &req.url);
            None
        }
        Err(err) => {
            eprintln!("Error fetching URL {}: {} ", &req.url, err);
            None
        }
    };
    (client, req, result)
}

/// Outcome from parsing the HTML.
enum ExtractionOutcome {
    Finished(Option<String>),         // either one candidate or no candidate
    NeedsUser(Vec<(String, String)>), // multiple candidates: list of (hanja, english) options
}

/// A message that requires user interaction.
struct UserInteraction {
    index: usize,
    hangul: String,
    eng_word: String,
    options: Vec<(String, String)>,
}

/// Reads a TSV file (skipping lines starting with '#') and returns rows.
fn read_tsv_file<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<String>>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .comment(Some(b'#'))
        .from_reader(BufReader::new(file));
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.len() >= (COLUMN_HANGUL.max(COLUMN_ENGLISH) + 1) {
            rows.push(record.iter().map(|s| s.to_string()).collect());
        }
    }
    Ok(rows)
}

fn write_tsv_file<P: AsRef<Path>>(path: P, rows: &[Vec<String>]) -> Result<()> {
    let file = File::create(path)?;
    let mut wtr = WriterBuilder::new().delimiter(b'\t').from_writer(file);
    for row in rows {
        wtr.write_record(row)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Clean up a string: remove newlines, tabs, extra spaces and some prefixes/suffixes.
fn clean_str(s: &str) -> String {
    let mut result = s
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .trim()
        .to_string();
    if result.ends_with("(conjugate verb)") {
        let cut = "(conjugate verb)";
        result = result[..result.len().saturating_sub(cut.len())]
            .trim()
            .to_string();
        result = clean_str(&result);
    } else if result.starts_with("a ") {
        result = result[2..].to_string();
    } else if result.starts_with("an ") {
        result = result[3..].to_string();
    } else if result.starts_with("one ") || result.starts_with("the ") {
        result = result[4..].to_string();
    }
    result
}

/// Constructs the URL given a hangul string.
fn map_to_url(hangul: &str) -> String {
    format!("https://koreanhanja.app/{}", hangul)
}

/// The fetch scheduler function. It owns a single receiver (`fetch_request_receiver`)
/// and manages up to 10 concurrent fetch jobs, including re-queuing failed ones
/// with a backoff timeout. It sends completed fetch results (successful or not)
/// on the `fetch_result_sender` channel.
///
/// Currently, this method will fail if there is a permanent fault i.e. no internet or response is 404
pub async fn fetch_scheduler(
    mut fetch_request_receiver: Receiver<FetchRequest>,
    client: ClientWithMiddleware,
    fetch_result_sender: Sender<FetchResult>,
) {
    let mut finished = 0;
    
    // Backoff intervals for re-queueing failed jobs.
    let backoff_intervals = [100, 500, 1000, 2000, 5000, 10000].map(Duration::from_millis);
    let mut current_backoff = -1isize;
    const MAX_CONCURRENT: usize = 10;

    // Pre-create 10 clients.
    let mut available_clients: Vec<ClientWithMiddleware> =
        (1..MAX_CONCURRENT).map(|_| client.clone()).collect();
    available_clients.push(client);

    // Active fetch jobs.
    let mut active_tasks: FuturesUnordered<_> = FuturesUnordered::new();

    // Pending jobs (failed or waiting for backoff). We use a VecDeque to requeue in order.
    let mut pending_jobs: VecDeque<PendingJob> = VecDeque::new();

    // A helper future for backoff: we create one if pending_jobs is nonempty.
    let mut backoff_sleep = None;

    while !(fetch_request_receiver.is_closed()
        && active_tasks.is_empty()
        && pending_jobs.is_empty())
    {
        let max_concurrent = if current_backoff < 0 {
            MAX_CONCURRENT
        } else {
            1
        };
        // If we have available clients and pending jobs, launch them.

        while backoff_sleep.is_none()
            && active_tasks.len() < max_concurrent
            && !available_clients.is_empty()
        {
            let Some(job) = pending_jobs.pop_front() else {
                break;
            };
            let client = available_clients.pop().unwrap();
            active_tasks.push(Box::pin(run_fetch_job(client, job.req)));
        }

        // If we have available clients and no active tasks, try to fetch new requests.
        while backoff_sleep.is_none()
            && active_tasks.len() < max_concurrent
            && !available_clients.is_empty()
        {
            let Ok(req) = fetch_request_receiver.try_recv() else {
                break;
            };
            let client = available_clients.pop().unwrap();
            active_tasks.push(Box::pin(run_fetch_job(client, req)));
        }

        // Now await one of several events.
        tokio::select! {
            biased;
            // (a) A new fetch request arrives.
            maybe_req = fetch_request_receiver.recv(), if active_tasks.len() < max_concurrent && !available_clients.is_empty() && pending_jobs.is_empty()  =>  {
                match maybe_req {
                    Some(req) => {
                        let client = available_clients.pop().unwrap();
                        active_tasks.push(Box::pin(run_fetch_job(client, req)));
                    }
                    None => {

                    }
                }
            },

            // (b) One active fetch task finished. We need to take the permit first, because otherwise a message may be lost
            Ok(permit) = fetch_result_sender.reserve(), if !active_tasks.is_empty() => {
                if let Some((client, req, response)) = active_tasks.next().await {
                    // Return the client to the pool.
                    available_clients.push(client);
                    if let Some(resp) = response {
                        backoff_sleep = None;
                        current_backoff = -1;
                        finished += 1;

                        aprintln!("Fetch req for {} / {} succeeded and was queued for processing. finish={} index={}", req.hangul, req.eng_word, req.index, finished);
                        // Success: send result.
                        let _ = permit.send(FetchResult {
                            index: req.index,
                            hangul: req.hangul,
                            eng_word: req.eng_word,
                            response: Some(resp),
                        });
                    } else {
                        aprintln!("Fetch req for {} / {} failed", req.hangul, req.eng_word);
                        // We failed - start backoff is not active, or if elapsed increate backoff
                        pending_jobs.push_back(PendingJob {
                            req
                        });


                        //backoff_sleep clears itself if the time elapsed or a task succeeded
                        if backoff_sleep.is_none() {
                            current_backoff += 1;
                            let now = TokioInstant::now();
                            let (backoff_interval, delta) = if current_backoff < backoff_intervals.len() as isize {
                                (backoff_intervals[current_backoff.unsigned_abs()], backoff_intervals[max(0, current_backoff-1).unsigned_abs()])
                            } else {
                                let max_backoff = *backoff_intervals.last().unwrap();
                                (max_backoff.mul((current_backoff as usize - backoff_intervals.len() + 1) as u32), max_backoff)
                            };
                            let offset = delta.mul_f32(random::<f32>()*2.);

                            aprintln!("Starting backoff at level {} will wait {:?}ms", current_backoff, backoff_interval + delta - offset);
                            backoff_sleep = Some(sleep_until(now + (backoff_interval + delta - offset)));
                        }
                    }
                }
            },
            // (c) Backoff timeout finished.
            // This branch is only active if we set backoff_sleep.
            Some(_) = conditional_sleeper(&mut backoff_sleep) => {
                //backoff_sleep clears itself if the time elapsed or a task succeeded
                aprintln!("backoff sleeper finished");
            },
            // (d) If no other branch is ready, sleep briefly.
            else => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            },
        }
    }

    aprintln!("Fetch scheduler exited.");
}

async fn conditional_sleeper(sleep: &mut Option<tokio::time::Sleep>) -> Option<()> {
    // match sleep.take_if(|s| s.is_elapsed()) {
    match sleep.take() {
        Some(timer) => Some(timer.await),
        None => None,
    }
}

/// Parses the HTML to extract similar words. If exactly one (or zero) candidate is found,
/// returns ExtractionOutcome::Finished. Otherwise, returns NeedsUser with options.
async fn extract_similar_words(
    resp: Response,
    hangul: &str,
    eng_word: &str,
) -> Result<ExtractionOutcome> {
    let text = resp.text().await?;
    let document = Html::parse_document(&text);
    let table_selector =
        Selector::parse("table.similar-words").map_err(|err| anyhow!(err.to_string()))?;
    let row_selector = Selector::parse("tr").map_err(|err| anyhow!(err.to_string()))?;
    let cell_selector = Selector::parse("td").map_err(|err| anyhow!(err.to_string()))?;

    let mut similar_words = Vec::new();

    if let Some(table) = document.select(&table_selector).next() {
        for row in table.select(&row_selector) {
            let cells: Vec<_> = row.select(&cell_selector).collect();
            if cells.len() == 3 {
                // Get the text content from each <td>, trim and clean it.
                let cell_texts: Vec<String> = cells
                    .iter()
                    .map(|cell| clean_str(&cell.text().collect::<Vec<_>>().join(" ")))
                    .collect();
                // Check if the second column matches the given hangul.
                if cell_texts[1] == hangul {
                    similar_words.push((cell_texts[0].clone(), cell_texts[2].clone()));
                }
            }
        }
    }

    // Remove duplicate hanja options (keep the first occurrence).
    similar_words.sort_by(|a, b| a.0.cmp(&b.0));
    similar_words.dedup_by(|a, b| a.0 == b.0);

    if similar_words.len() <= 1 {
        if similar_words.len() == 0 {
            aprintln!("parsed result for {} and found no hanja", hangul);
        } else {
            aprintln!(
                "parsed result for {} and got {:?}",
                hangul,
                similar_words[0]
            );
        }
        // either one candidate or none
        Ok(ExtractionOutcome::Finished(
            similar_words.first().map(|(h, _)| h.clone()),
        ))
    } else {
        aprintln!(
            "parsed result for {} and found multiple hanja {:?}",
            hangul,
            similar_words
        );
        Ok(ExtractionOutcome::NeedsUser(similar_words))
    }
}

/*async fn run_not_send<F>(future: F) -> anyhow::Result<F::Output>
    where
    F: Future,
    F::Output: Send,
{
    tokio::task::spawn_blocking(move || {
        let local = tokio::task::LocalSet::new();
        local.run_until(async move {
            match tokio::task::spawn_local(async move {
                let x = future.await;
                x
            }).await {
                Ok(v) => Ok(v),
                Err(_) => Err(())
            }
        })
    }).await?
}*/

/// Uses dialoguer in a blocking thread to prompt the user for a choice.
async fn prompt_user(
    term: &mut TokioChannelConsoleAccess,
    hangul: String,
    eng_word: String,
    options: Vec<(String, String)>,
) -> Option<String> {
    aprintln!(
        "\nMultiple options for hangul: {} and english: {}",
        hangul,
        eng_word
    );
    let mut items = vec!["[s] Skip".to_string()];
    for (i, (hanja, eng)) in options.iter().enumerate() {
        items.push(format!("[{}] Hanja: {}, English: {}", i, hanja, eng));
    }

    let selection = Select::new()
        .with_prompt("Select an option")
        .items(&items)
        .default(0)
        .interact_on_async(term)
        .await
        .unwrap_or(0);

    if selection == 0 {
        None
    } else {
        Some(options[selection - 1].0.clone())
    }
}

static STD_OUT_SENDER_GLOBAL: OnceLock<UnboundedSender<String>> = OnceLock::new();
thread_local! {
    static STD_OUT_SENDER: UnboundedSender<String> = STD_OUT_SENDER_GLOBAL.get().unwrap().clone();
}

fn _println_async(line: String) {
    _ = STD_OUT_SENDER.with(|sender| sender.send(line));
}

static STDIN_TASK_SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
#[tokio::main]
async fn main() -> Result<()> {
    let (stdout_sender, stdout_receiver) = mpsc::unbounded_channel::<String>();
    let (stdin_sender, stdin_receiver) = mpsc::channel::<Key>(100);
    let term_access = TokioChannelConsoleAccess {
        term: Term::stdout(),
        key_receiver: stdin_receiver,
        write_line_receiver: stdout_receiver,
    };
    _ = STD_OUT_SENDER_GLOBAL.set(stdout_sender);
    let _stdin_task = tokio::task::spawn(async move {
        let t = Term::stdout();
        while !(&STDIN_TASK_SHUTDOWN_REQUESTED).load(Ordering::Relaxed) {
            let Ok(permit) = stdin_sender.reserve().await else {
                eprint!("stdin thread channel was closed, exiting: ");
                return;
            };
            let key = match tokio::task::block_in_place(|| t.read_key()) {
                Ok(k) => k,
                Err(err) => {
                    eprint!("stdin thread received error, exiting: {:?}", err);
                    return;
                }
            };
            permit.send(key);
        }
    });

    // Read TSV file and build word pairs.
    let mut rows = read_tsv_file(INPUT_FILE)?;
    aprintln!("Read {} rows from {}", rows.len(), INPUT_FILE);

    // Create a vector of word pairs from the rows.
    // We assume COLUMN_HANGUL and COLUMN_ENGLISH are present.
    let word_pairs: Vec<(String, String)> = rows
        .iter()
        .map(|row| (row[COLUMN_HANGUL].clone(), row[COLUMN_ENGLISH].clone()))
        .collect();

    // Build a reqwest client with middleware that caches responses to "./.cache"
    let client = ClientBuilder::new(reqwest::Client::new())
        .with(Cache(HttpCache {
            mode: CacheMode::Default,
            manager: CACacheManager::default(),
            options: HttpCacheOptions::default(),
        }))
        .build();

    // --- Create channels ---
    let (fetch_request_sender, fetch_request_receiver) = mpsc::channel::<FetchRequest>(100);
    let (fetch_result_sender, fetch_result_receiver) = mpsc::channel::<FetchResult>(100);
    let (commited_result_sender, mut commited_result_receiver) =
        mpsc::channel::<(usize, Option<String>)>(100);
    let (user_int_sender, mut user_int_receiver) = mpsc::channel::<UserInteraction>(100);

    // --- Spawn fetch scheduler ---
    tokio::spawn(fetch_scheduler(
        fetch_request_receiver,
        client,
        fetch_result_sender,
    ));
    // Note: if mpsc::Receiver does not support multiple receivers, you may instead spawn tasks that share a single receiver via Arc<Mutex<_>>.

    // --- Spawn extraction task ---
    tokio::spawn(process_parsing_result(
        user_int_sender,
        commited_result_sender.clone(),
        fetch_result_receiver,
    ));

    // --- Spawn user interaction task ---
    tokio::spawn(perform_user_interactions(
        term_access,
        user_int_receiver,
        commited_result_sender.clone(),
    ));

    // --- Spawn aggregator task ---
    let total = word_pairs.len();
    // Create a container for final results.
    let mut results: HashMap<usize, Option<String>> = HashMap::new();
    let aggregator = tokio::spawn(async move {
        while let Some((index, result)) = commited_result_receiver.recv().await {
            aprintln!("finished processing index {} and got {:?}, missing {} replies", index, result, total - results.len());
            results.insert(index, result);
            if results.len() == total {
                break;
            }
        }
        results
    });

    // --- Send all fetch requests ---
    {
        for (index, (hangul, eng)) in word_pairs.iter().enumerate() {
            let req = FetchRequest {
                index,
                hangul: hangul.clone(),
                eng_word: eng.clone(),
                url: map_to_url(hangul),
            };
            if fetch_request_sender.send(req).await.is_err() {
                eprintln!("Failed to send fetch request for index {}", index);
            }
        }
    }
    // Dropping fetch_req_tx will signal the fetcher tasks to eventually exit.

    // Await the aggregator to get final results.
    let final_results = aggregator.await?;
    // Build the updated rows: append the fetched hanja value as a new column.
    for (i, row) in rows.iter_mut().enumerate() {
        let hanja = final_results
            .get(&i)
            .cloned()
            .unwrap_or(None)
            .unwrap_or_default();
        row.push(hanja);
    }
    write_tsv_file(OUTPUT_FILE, &rows)?;
    aprintln!("Wrote output to {}", OUTPUT_FILE);

    // In a real application you would join on all spawned tasks.
    // Here we simply await the aggregator.
    STDIN_TASK_SHUTDOWN_REQUESTED.swap(true, Ordering::Relaxed);
    Ok(())
}

async fn perform_user_interactions(
    mut term_access: TokioChannelConsoleAccess,
    mut user_int_receiver: Receiver<UserInteraction>,
    commited_result_sender: Sender<(usize, Option<String>)>,
) {
    loop {
        select! {
            biased;
            Some(line) = term_access.write_line_receiver.recv() => {
                println!("{}", line)
            }
            Some(ui) = user_int_receiver.recv() => {
                println!("prompting user for: {}", ui.hangul);
                if let Some(chosen) =
                    prompt_user(&mut term_access, ui.hangul.clone(), ui.eng_word.clone(), ui.options).await
                {
                    let _ = commited_result_sender.send((ui.index, Some(chosen))).await;
                } else {
                    let _ = commited_result_sender.send((ui.index, None)).await;
                }
            }
            else => return
        }
    }
}

async fn process_parsing_result(
    user_int_sender: Sender<UserInteraction>,
    commited_result_sender: Sender<(usize, Option<String>)>,
    mut fetch_result_receiver: Receiver<FetchResult>,
) {
    while let Some(fetch_res) = fetch_result_receiver.recv().await {
        if let Some(response) = fetch_res.response {
            match extract_similar_words(response, &fetch_res.hangul, &fetch_res.eng_word).await {
                Ok(ExtractionOutcome::Finished(result)) => {
                    let _ = commited_result_sender.send((fetch_res.index, result)).await;
                }
                Ok(ExtractionOutcome::NeedsUser(options)) => {
                    aprintln!(
                        "queuing {} / {} ({}) for user interaction:",
                        fetch_res.hangul,
                        fetch_res.index,
                        fetch_res.eng_word
                    );
                    let ui = UserInteraction {
                        index: fetch_res.index,
                        hangul: fetch_res.hangul,
                        eng_word: fetch_res.eng_word,
                        options,
                    };
                    let _ = user_int_sender.send(ui).await;
                }
                Err(e) => {
                    eprintln!("Error in extraction for index {}: {:?}", fetch_res.index, e);
                    let _ = commited_result_sender.send((fetch_res.index, None)).await;
                }
            }
        } else {
            // No response: send an empty result.
            let _ = commited_result_sender.send((fetch_res.index, None)).await;
        }
    }
}
