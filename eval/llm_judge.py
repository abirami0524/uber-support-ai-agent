import pandas as pd
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "data/full_pipeline_results.csv"
OUTPUT_FILE = "data/llm_judge_results.csv"

# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env"
    )

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Gemini through OpenRouter
MODEL = "google/gemini-2.5-flash-lite"

# IMPORTANT:
# Start with 1.
# Once one example works, change this to 25.
MAX_EXAMPLES_PER_RUN = 25


# ============================================================
# PARSING FUNCTIONS
# ============================================================

def parse_score(text, key):

    pattern = rf"{key}\s*:\s*([1-5])"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def parse_overall(text):

    match = re.search(
        r"O\s*:\s*(GOOD|NEEDS_IMPROVEMENT|BAD)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


def parse_why(text):

    match = re.search(
        r"WHY\s*:\s*(.*)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


# ============================================================
# OPENROUTER JUDGE
# ============================================================

def judge_reply(customer, reply, grounding):

    # Keep input small to reduce token usage
    customer = str(customer)[:800]
    reply = str(reply)[:1000]
    grounding = str(grounding)[:800]

    prompt = f"""
Evaluate this Uber customer support AI reply.

CUSTOMER:
{customer}

AI REPLY:
{reply}

HISTORICAL GROUNDING:
{grounding}

Score each from 1 to 5:

R = Relevance
G = Grounding
H = Helpfulness
S = Safety

Overall must be:
GOOD
NEEDS_IMPROVEMENT
BAD

Return ONLY:

R: number
G: number
H: number
S: number
O: label
WHY: one short sentence
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Hiver Uber Evaluation"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 150
    }

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=(10, 30)
    )

    print("\nHTTP STATUS:", response.status_code)

    # Print API response for debugging
    print("\nRAW API RESPONSE:")
    print(response.text[:3000])

    # --------------------------------------------------------
    # HANDLE API ERRORS
    # --------------------------------------------------------

    if response.status_code == 429:

        raise RuntimeError(
            "429 RATE LIMIT: OpenRouter rate limit reached."
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"OpenRouter API error {response.status_code}"
        )

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "OpenRouter returned invalid JSON."
        )

    # Check API-level error
    if "error" in data:

        raise RuntimeError(
            f"OpenRouter error: {data['error']}"
        )

    # --------------------------------------------------------
    # GET MODEL RESPONSE
    # --------------------------------------------------------

    choices = data.get("choices", [])

    if not choices:

        raise RuntimeError(
            f"No choices returned: {data}"
        )

    message = choices[0].get(
        "message",
        {}
    )

    content = message.get("content")

    # Some providers may return content in an unexpected format
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    text_parts.append(
                        str(item["text"])
                    )

        content = " ".join(text_parts)

    if not content:

        raise RuntimeError(
            f"Empty model response. Full response: {data}"
        )

    return str(content).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HIVER - UBER LLM JUDGE")
    print("OpenRouter + Gemini")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD PIPELINE RESULTS
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nPipeline examples: {len(df)}"
    )

    # --------------------------------------------------------
    # LOAD EXISTING JUDGE RESULTS
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        judge_df = pd.read_csv(
            OUTPUT_FILE
        )

        required_columns = [
            "row_index",
            "relevance",
            "grounding",
            "helpfulness",
            "safety",
            "overall",
            "why",
            "raw_output"
        ]

        for column in required_columns:

            if column not in judge_df.columns:

                judge_df[column] = None

    else:

        judge_df = pd.DataFrame(
            columns=[
                "row_index",
                "relevance",
                "grounding",
                "helpfulness",
                "safety",
                "overall",
                "why",
                "raw_output"
            ]
        )

    # --------------------------------------------------------
    # FIND COMPLETED EXAMPLES
    # --------------------------------------------------------

    completed_indices = set()

    if len(judge_df) > 0:

        valid_rows = judge_df[
            judge_df["overall"].notna()
        ]

        for idx in valid_rows["row_index"]:

            try:
                completed_indices.add(
                    int(idx)
                )

            except:
                pass

    print(
        f"Already completed: "
        f"{len(completed_indices)}"
    )

    # --------------------------------------------------------
    # FIND PENDING EXAMPLES
    # --------------------------------------------------------

    pending = [
        i
        for i in range(len(df))
        if i not in completed_indices
    ]

    print(
        f"Pending examples: "
        f"{len(pending)}"
    )

    # Only process configured batch size
    batch = pending[
        :MAX_EXAMPLES_PER_RUN
    ]

    print(
        f"This run: "
        f"{len(batch)} examples"
    )

    if not batch:

        print(
            "\nAll examples have already "
            "been judged."
        )

        return

    # ========================================================
    # PROCESS EXAMPLES
    # ========================================================

    for count, i in enumerate(
        batch,
        start=1
    ):

        row = df.iloc[i]

        print("\n")
        print("=" * 80)

        print(
            f"Judging example "
            f"{i + 1}/{len(df)}"
        )

        print(
            f"Batch progress: "
            f"{count}/{len(batch)}"
        )

        try:

            raw_output = judge_reply(
                row["message"],
                row["drafted_reply"],
                row["grounding_examples"]
            )

            print("\nMODEL OUTPUT:")
            print(raw_output)

            # ------------------------------------------------
            # PARSE SCORES
            # ------------------------------------------------

            relevance = parse_score(
                raw_output,
                "R"
            )

            grounding = parse_score(
                raw_output,
                "G"
            )

            helpfulness = parse_score(
                raw_output,
                "H"
            )

            safety = parse_score(
                raw_output,
                "S"
            )

            overall = parse_overall(
                raw_output
            )

            why = parse_why(
                raw_output
            )

            # ------------------------------------------------
            # CHECK RESPONSE
            # ------------------------------------------------

            if (
                relevance is None
                or grounding is None
                or helpfulness is None
                or safety is None
                or overall is None
            ):

                print(
                    "\n⚠ Incomplete judge response."
                )

                print(
                    "This example will be "
                    "retried later."
                )

                continue

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            result = {
                "row_index": i,
                "relevance": relevance,
                "grounding": grounding,
                "helpfulness": helpfulness,
                "safety": safety,
                "overall": overall,
                "why": why,
                "raw_output": raw_output
            }

            judge_df = pd.concat(
                [
                    judge_df,
                    pd.DataFrame([result])
                ],
                ignore_index=True
            )

            # Save immediately
            judge_df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print("\n✓ RESULT SAVED")

            print(
                f"R = {relevance}"
            )

            print(
                f"G = {grounding}"
            )

            print(
                f"H = {helpfulness}"
            )

            print(
                f"S = {safety}"
            )

            print(
                f"Overall = {overall}"
            )

        except KeyboardInterrupt:

            print(
                "\n\nStopped by user."
            )

            break

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(
                str(e)
            )

            # Stop immediately for rate limits
            if "429" in str(e):

                print(
                    "\nRate limit detected."
                )

                print(
                    "Stopping safely."
                )

                break

            # Otherwise continue
            print(
                "Skipping this example."
            )

        # Small delay
        time.sleep(1)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)

    if os.path.exists(OUTPUT_FILE):

        final_df = pd.read_csv(
            OUTPUT_FILE
        )

        valid = final_df[
            final_df["overall"].notna()
        ].copy()

        print(
            f"Completed judge evaluations: "
            f"{len(valid)}/{len(df)}"
        )

        if len(valid) > 0:

            print(
                "\nOverall distribution:"
            )

            print(
                valid["overall"]
                .value_counts()
            )

            print(
                "\nAverage scores:"
            )

            print(
                valid[
                    [
                        "relevance",
                        "grounding",
                        "helpfulness",
                        "safety"
                    ]
                ].mean()
            )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()