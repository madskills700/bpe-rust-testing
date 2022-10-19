use std::alloc::dealloc;
use std::fs;
use std::os::unix::raw::time_t;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Utc};
use serde_derive::Deserialize;
use serde_derive::Serialize;
use serde_json::Value;
use tokenizers::{AddedToken, Encoding, Model, Result, TokenizerBuilder};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{NormalizerWrapper, strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TgDump {
    pub name: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub id: i64,
    pub messages: Vec<Message>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: i64,
    #[serde(rename = "type")]
    pub type_field: String,
    pub date: Option<String>,
    pub from: Option<String>,
    #[serde(rename = "from_id")]
    pub from_id: Option<String>,
    pub photo: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub text: Value,
}

fn main() -> Result<()> {
    let start = Instant::now();
    println!("Start {}", Utc::now());
    let json = fs::read_to_string("resources/result.json")
        .expect("пустили жаба-обезъяну");
    let parsed: TgDump = serde_json::from_str(&json).unwrap();
    println!("Json parsed  {}", Utc::now());
    drop(json);
    let messages: Vec<String> = parsed.messages.iter().map(|it| it.text.to_string()).collect();
    println!("Mapped on string vector {}", Utc::now());
    drop(parsed);
    let dict_size = 30000;

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(dict_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<unk>"), true),
            AddedToken::from(String::from("<mask>"), true),
            AddedToken::from(String::from("<cls>"), true),
            AddedToken::from(String::from("<sep>"), true),
        ])
        .build();


    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let trained = tokenizer.train(&mut trainer, messages.iter());
    println!("Trained {}", Utc::now());
    let res = tokenizer.encode_batch(messages, true);
    println!("Success tokenizing! Execution time: {:?}", start.elapsed());
    Ok(())
}
