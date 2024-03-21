# Prompt Sequence JSON Specifications

- [Prompt Sequence JSON Specifications](#prompt-sequence-json-specifications)
  - [Specifications](#specifications)
  - [Usage](#usage)
  - [Examples](#examples)

---

## Specifications

```json
{
    "_v": "1.1.0",
    "defaults": {
        "positive": "a pet driving a vehicle",
        "negative": "people"
    },
    "sequence": [
        {
            "positive": "a black lab dog driving a cigar racer boat, happy, studio lighting",
            "negative": "people, cat, crocodile",
            "timecode": 0
        },
        {
            "positive": "a tuxedo cat smoking a pipe, sitting on a chair, happy, studio lighting",
            "negative": "people, dog, crocodile",
            "timecode": 3
        },
        {
            "positive": "a crocodile wearing a top hat and monocle, lounging on a recliner, happy, studio lighting",
            "negative": "people, cat, dog",
            "timecode": 7
        }
    ]
}
```
* `_v`: The version of the Prompt Sequence JSON file following [Semantic Versioning 2.0.0 specifications](https://semver.org).
* `defaults`: A set of default values to use for optional values in the `sequence`.
    * `positive`: A default positive prompt to be used if one is not provided.
    * `negative`: A default negative prompt to be used if one is not provided.
* `sequence`: A list of prompt data in order from top to bottom.
    * `positive`: _(Optional)_ A positive prompt for this item.
        * If not provided, the value from `defaults` will be used.
    * `negative`: _(Optional)_ A negative prompt for this item.
        * If not provided, the value from `defaults` will be used.
    * `timecode`: _(Optional)_ The time code in seconds where a prompt should reach its max influence.
        * If not provided, `-1` will be supplied to indicate no set time code.

Note that if you would like a prompt to use _only_ the defaults, provide an empty dictionary like `{}` at the appropriate indexes in your `sequence`.

## Usage

A Prompt Sequence JSON file can be loaded by the `Prompt Sequence Loader` node.

Prompt Sequences, once encoded, cannot currently be exported.

## Examples

A complete example with all optional values provided can be found above in the [Specifications](#specifications) section.

For a practical example that has appropriate exclusions for its task, view the [example.json](../promptSequences/example.json) file provided in this repo.
