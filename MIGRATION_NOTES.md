# Migration to google.genai & Quota Management

## Changes Made

### 1. Package Migration (google-generativeai → google-genai)

**Why:** The `google-generativeai` package has been deprecated and is no longer receiving updates or bug fixes.

**Changes:**

- Updated imports from `google.generativeai` to `google.genai`
- Changed API usage from `genai.GenerativeModel()` to `gemini_client.models.generate_content()`
- Updated [requirements.txt](requirements.txt) to use `google-genai` instead of deprecated package

### 2. Quota Management & Throttling

Added intelligent request handling to prevent quota exceeded errors:

**Features:**

- **Request Caching**: Responses are cached to avoid duplicate API calls for identical queries
- **Rate Limiting**: Enforces minimum 2-second interval between requests to avoid rate limits
- **Quota Detection**: Automatically detects 429 errors (quota exceeded) and pauses requests
- **Retry Logic**: Extracts retry delay from API response and pauses until quota resets
- **User Feedback**: Informs users when quota is exceeded instead of silent failures

### 3. Error Handling Improvements

- Better detection and parsing of quota errors
- Graceful degradation with informative messages
- Automatic pause when hitting quotas instead of continuous failed requests

## Implementation Details

```python
# New throttling configuration in AIBot.__init__()
self.request_cache = {}              # Cache responses
self.last_request_time = defaultdict(float)  # Track timing
self.quota_exceeded_until = 0        # When quota resets
self.min_request_interval = 2        # Minimum seconds between requests
```

## API Changes

### Old (Deprecated)

```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')
response = gemini_model.generate_content(prompt)
```

### New

```python
import google.genai as genai
genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
response = gemini_client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt
)
```

## Migration Checklist

- [x] Update package imports
- [x] Update API calls to use new google.genai interface
- [x] Add request throttling mechanism
- [x] Add response caching
- [x] Implement quota detection and waiting
- [x] Update requirements.txt
- [x] Improve error messaging

## Deployment Notes

1. Install the new package: `pip install google-genai`
2. Remove old package: `pip uninstall google-generativeai` (optional)
3. Deploy updated code
4. Monitor logs for quota improvements

## Expected Benefits

✅ No more "deprecated package" warnings  
✅ Reduced quota exceeded errors through caching  
✅ Better request rate management  
✅ Improved user experience with informative messages  
✅ Automatic pause/resume on quota limit

## References

- [google-genai GitHub](https://github.com/google/generative-ai-python)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
