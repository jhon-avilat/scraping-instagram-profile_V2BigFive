# Instagram Profile Analyzer (v3GeminiAPI)

An academic project that uses **Playwright** to collect data from public Instagram profiles and the **Gemini API** to generate an exploratory personality estimate based on the Big Five model.

The project includes a local web dashboard to scrape profiles, explore posts, analyze personality traits and compare the results with a short self-report questionnaire (BFI-10).

## Features

- Collects profile information, posts, reels and image descriptions.
- Extracts captions, likes, comments and dates from selected posts.
- Uses Gemini API to estimate the five Big Five personality traits.
- Runs the analysis multiple times to evaluate score consistency.
- Anonymizes usernames and text before sending data to Gemini.
- Includes a BFI-10 questionnaire to compare the estimate with self-reported results.
- Saves results locally as JSON files.

## Requirements

- Python 3.9 or newer
- A secondary Instagram account
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
git clone https://github.com/jhon-avilat/instagram-profile-analyzer-v3GeminiAPI.git
cd instagram-profile-analyzer-v3GeminiAPI

python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Install dependencies and run the project:

```bash
pip install -r requirements.txt
playwright install chromium
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Configuration

### Instagram

Instagram may require an authenticated session to access profile content. The recommended option is to log in manually using the browser opened by Playwright. The session is saved in `browser_profile/` and reused on future runs.

Use a secondary test account and avoid sharing your session cookies.

### Gemini API

Create an API key in [Google AI Studio](https://aistudio.google.com/apikey) and add it to a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

The default model is `gemini-flash-latest`. You can change it using `GEMINI_MODEL` in `.env`.

Check the available models with:

```bash
python analyzer.py --list-models
```

## How it works

1. **Data collection:** Playwright controls Chromium to navigate Instagram, collect profile information and extract post details.
2. **Data processing:** The scraper removes real commenter usernames and replaces them with anonymous labels.
3. **Privacy:** Before sending data to Gemini, the analyzer removes usernames, URLs and other identifying information.
4. **Personality analysis:** Gemini returns scores from 1 to 5 for each Big Five trait, along with confidence, evidence and explanations.
5. **Consistency:** The analysis runs multiple times and compares the results to identify how stable the estimates are.
6. **Validation:** The user can complete the BFI-10 questionnaire and compare its results with Gemini's estimates.

The application uses **Flask** for the backend, HTML/CSS and vanilla JavaScript for the dashboard, Playwright for browser automation, and JSON files for local storage.

## The Big Five

The model describes personality through five continuous traits:

| Trait | What it represents |
|---|---|
| Openness | Curiosity, imagination and openness to new experiences |
| Conscientiousness | Organization, discipline and responsibility |
| Extraversion | Sociability, energy and assertiveness |
| Agreeableness | Cooperation, trust and empathy |
| Neuroticism | Tendency to experience worry, stress and negative emotions |

Scores are not good or bad, and they do not define a person. The results are exploratory estimates, not psychological assessments.

## Limitations and responsible use

- Personality estimates based on social media posts are uncertain and should not be treated as accurate measurements.
- Analyze your own profile or a profile whose owner has given informed consent.
- The dashboard requires consent confirmation, but cannot independently verify it.
- Although identifying information is removed before sending text to Gemini, the collected data and resulting personality estimates should still be treated as sensitive.
- Instagram may restrict automated access, and changes to its website can break the scraper. Keep collection limited and respect the platform's terms.
- Data protection regulations, including Ecuador's LOPDP, may apply.
- For accounts you manage, consider using the official [Instagram Graph API](https://developers.facebook.com/docs/instagram-platform).

## Reference

Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory in English and German. *Journal of Research in Personality, 41*(1), 203–212.

## Author

Built by [Jhon Avila](https://github.com/jhon-avilat) as an academic project.

Released under the [MIT License](LICENSE).