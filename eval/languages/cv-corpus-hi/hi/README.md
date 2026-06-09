# *हिंदी* &mdash; Hindi (`hi`)

This datasheet is for cv-corpus-25.0-2026-03-09 of the Mozilla Common Voice *Scripted Speech* dataset for Hindi [हिंदी - `hi`]. The dataset contains 19006 clips representing 26.63 hours of recorded speech (15.56 hours validated) from 474 speakers, recorded from a text corpus of 42,169 sentences.

## Language

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 2,952 (15.5%) | 41 (8.6%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 9,515 (50.1%) | 129 (27.2%) |
| female_feminine | Female, feminine | 571 (3.0%) | 21 (4.4%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | - | - |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 8,920 (46.9%) | 338 (71.3%) |

*Gender declared: 10,086 of 19,006 clips (53.1%), 136 of 474 speakers (28.7%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 184 (1.0%) | 16 (3.4%) |
| twenties | Twenties | 5,220 (27.5%) | 100 (21.1%) |
| thirties | Thirties | 6,081 (32.0%) | 31 (6.5%) |
| fourties | Fourties | 1,723 (9.1%) | 16 (3.4%) |
| fifties | Fifties | 255 (1.3%) | 2 (0.4%) |
| sixties | Sixties | 99 (0.5%) | 2 (0.4%) |
| seventies | Seventies | - | - |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 5,444 (28.6%) | 323 (68.1%) |

*Age declared: 13,562 of 19,006 clips (71.4%), 151 of 474 speakers (31.9%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 11,108 (58.4%) |
| Invalidated | 946 (5.0%) |
| Other | 6,952 (36.6%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 4,894 (44.1%) |
| Dev | 2,809 (25.3%) |
| Test | 3,326 (29.9%) |

*Training split coverage: 11,029 of 11,108 validated clips (99.3%)*

The dataset contains 11108 validated, 946 invalidated, and 6952 unresolved clips. The average clip duration is 5.045 seconds.

## Text corpus

**Validated sentences:** 32,204

| Category | Count |
|---|---|
| Unvalidated sentences | 9,965 |
| Pending sentences | 9,958 |
| Rejected sentences | 7 |
| Reported sentences | 146 |

The corpus contains 42,169 sentences: 32,204 validated and 9,965 unvalidated (9,958 pending review, 7 rejected), with 146 reported for review.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *प्रियंका-निक की क्रिश्चियन वेडिंग डेट पर सस्पेंस, ऐसी है चर्चा*
2. *'लंदन ओलंपिक में ध्वजवाहक बनने के हकदार सुशील'*
3. *इन आदतों के चलते ही प्रभावित हो रही है आपकी नींद*
4. *परमाणु प्रसार में कभी शामिल नहीं रहे: चीन*
5. *दीपिका ने नहीं बनवाया रणबीर के नाम का टैटू*

### Sources

| Source | Sentences |
|---|---|
| sentence-collector | 32,033 (99.5%) |
| Other | 171 (0.5%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `text` - supposed transcription of the audio
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here
- `prompt_upvotes` - number of upvotes the sentence prompt received
- `prompt_reports` - number of reports the sentence prompt received
- `is_edited` - whether the clip's transcription has been edited

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/hi/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/hi/speak)
- [Write](https://commonvoice.mozilla.org/hi/write)
- [Listen](https://commonvoice.mozilla.org/hi/listen)
- [Review](https://commonvoice.mozilla.org/hi/review)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
