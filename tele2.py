import os
import logging
import re
import asyncio
import time
from collections import defaultdict
import google.genai as genai
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,  # Add this import
    filters, 
    ContextTypes
)

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize clients - google.genai uses API key directly in Client()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

class PatternRecognitionHandler:
    def __init__(self):
        self.transformation_types = {
            'letter_shift': self._check_letter_shift,
            'rearrangement': self._check_rearrangement,
            'position_swap': self._check_position_swap,
            'letter_replacement': self._check_replacement
        }

    def analyze_pattern(self, question):
        pairs = re.findall(r'([A-Z]+)\s*:\s*([A-Z]+)', question)
        if not pairs:
            return None

        analysis = []
        for source, target in pairs:
            patterns = []
            for pattern_type, checker in self.transformation_types.items():
                if result := checker(source, target):
                    patterns.append((pattern_type, result))
            analysis.append((source, target, patterns))

        return self._format_pattern_analysis(analysis)

    def _check_letter_shift(self, source, target):
        if len(source) != len(target):
            return None
        shifts = []
        for s, t in zip(source, target):
            shift = (ord(t) - ord(s)) % 26
            shifts.append(shift)
        return shifts if len(set(shifts)) <= 2 else None

    def _check_rearrangement(self, source, target):
        return sorted(source) == sorted(target)

    def _check_position_swap(self, source, target):
        if len(source) != len(target):
            return None
        swaps = []
        for i, (s, t) in enumerate(zip(source, target)):
            if s != t:
                swaps.append((i, target.index(s)))
        return swaps if swaps else None

    def _check_replacement(self, source, target):
        if len(source) != len(target):
            return None
        replacements = {}
        for s, t in zip(source, target):
            if s != t:
                replacements[s] = t
        return replacements if replacements else None

    def _format_pattern_analysis(self, analysis):
        response = "🔍 Pattern Analysis:\n═══════════════════\n\n"
        
        for source, target, patterns in analysis:
            response += f"📌 {source} ➡️ {target}:\n"
            for pattern_type, details in patterns:
                if pattern_type == 'letter_shift':
                    response += "   • Letter shifting pattern detected\n"
                    response += f"     Shifts: {details}\n"
                elif pattern_type == 'rearrangement':
                    response += "   • Letters are rearranged\n"
                elif pattern_type == 'position_swap':
                    response += "   • Position swapping detected\n"
                    response += f"     Swaps: {details}\n"
                elif pattern_type == 'letter_replacement':
                    response += "   • Letter replacement pattern\n"
                    response += f"     Replacements: {details}\n"
            response += "\n"

        similar_pairs = self._find_similar_patterns(analysis)
        if similar_pairs:
            response += "✨ Similar Transformations Found:\n"
            for pair in similar_pairs:
                response += f"• {pair[0]} and {pair[1]} share the same pattern\n"
        else:
            response += "❗ No two pairs share exactly the same transformation pattern\n"

        return response

    def _find_similar_patterns(self, analysis):
        similar = []
        for i in range(len(analysis)):
            for j in range(i + 1, len(analysis)):
                if self._compare_patterns(analysis[i][2], analysis[j][2]):
                    similar.append((analysis[i][0], analysis[j][0]))
        return similar

    def _compare_patterns(self, pattern1, pattern2):
        if len(pattern1) != len(pattern2):
            return False
        return all(p1[0] == p2[0] and p1[1] == p2[1] for p1, p2 in zip(pattern1, pattern2))

class MathHandler:
    def solve(self, expression):
        try:
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            return eval(safe_expr, {"__builtins__": {}}, {})
        except:
            return None

class AptitudeHandler:
    def __init__(self):
        self.patterns = {
            'percentage': r'(\d+(\.\d+)?%|\bpercent\b)',
            'profit_loss': r'\b(profit|loss)\b',
            'time_distance': r'\b(speed|time|distance)\b',
            'ratio': r'\b(ratio|proportion)\b',
            'average': r'\b(average|mean)\b',
            'sequence': r'\b(sequence|series|next number)\b'
        }

    def detect_type(self, question):
        for qtype, pattern in self.patterns.items():
            if re.search(pattern, question.lower()):
                return qtype
        return None

# Add these imports at the top
import io
from PIL import Image
import pytesseract

# After existing imports
# Configure pytesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AIBot:
    def __init__(self):
        self.aptitude = AptitudeHandler()
        self.math = MathHandler()
        self.pattern_recognition = PatternRecognitionHandler()
        self.allowed_group_ids = [-1001369278049]
        self.programming_questions = {}
        self.gemini_config = {
            'temperature': 0.3,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 4096,
        }
        self.is_active = True
        # Add request throttling and caching
        self.request_cache = {}  # Cache for responses
        self.last_request_time = defaultdict(float)  # Track request times per user
        self.quota_exceeded_until = 0  # Track when quota will reset
        self.min_request_interval = 2  # Minimum seconds between requests
        
    async def should_respond(self, chat_id, message_text):
        if not message_text or message_text.startswith('/'):
            return False
        return chat_id in self.allowed_group_ids and self.is_active  # Modify this line

    async def analyze_image(self, image_file):
        """Analyze image content and provide a solution"""
        try:
            # Read image bytes
            image_bytes = image_file.read()
            
            prompt = """Analyze this image carefully. If it contains:
            - Text: Extract and read all text
            - Math problems: Provide step-by-step solutions
            - Code: Explain the code and suggest improvements
            - Questions: Provide detailed answers
            
            Format your response clearly and provide detailed explanations."""
            
            # Get response from Gemini with image
            response = await asyncio.to_thread(
                self._analyze_image_with_genai, image_bytes, prompt
            )
            
            if response:
                return self.clean_response(response)
            else:
                return "❌ I couldn't analyze this image. Please try with a clearer image."
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return "❌ I encountered an error analyzing this image. Please try again with a clearer image."
    
    def _analyze_image_with_genai(self, image_bytes, prompt):
        """Analyze image using google.genai API with fallback"""
        try:
            import base64
            # Encode image as base64
            image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            
            # Create image part for the API
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_base64
            }
            
            # Try gemini-2.5-pro first
            logger.info("Attempting image analysis with gemini-2.5-pro...")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    prompt,
                    image_part
                ]
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            elif response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    return ' '.join(part.text for part in parts if hasattr(part, 'text'))
            return None
            
        except Exception as e:
            error_str = str(e).lower()
            # If quota exceeded with pro, fall back to flash
            if "429" in str(e) or "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"gemini-2.5-pro image quota exceeded, falling back to gemini-2.5-flash")
                try:
                    import base64
                    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")
                    image_part = {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                    
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            prompt,
                            image_part
                        ]
                    )
                    
                    if response and hasattr(response, 'text'):
                        return response.text
                    elif response and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            return ' '.join(part.text for part in parts if hasattr(part, 'text'))
                    return None
                except Exception as flash_error:
                    logger.error(f"gemini-2.5-flash image analysis also failed: {flash_error}")
                    raise flash_error
            else:
                logger.error(f"_analyze_image_with_genai error: {e}")
                raise

    async def get_gemini_response(self, prompt):
        try:
            # Check cache first
            cache_key = str(prompt)[:100]  # Use first 100 chars as key
            if cache_key in self.request_cache:
                logger.info("Using cached response")
                return self.request_cache[cache_key]
            
            # Check if quota exceeded and wait
            current_time = time.time()
            if current_time < self.quota_exceeded_until:
                wait_time = int(self.quota_exceeded_until - current_time)
                return f"⏳ API quota exceeded. Please try again in {wait_time} seconds."
            
            # Check request interval
            if current_time - self.last_request_time["global"] < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval)
            
            self.last_request_time["global"] = time.time()
            
            # Handle both text and multimodal prompts
            response = await asyncio.to_thread(
                self._generate_with_genai, prompt
            )
            
            if response:
                # Cache successful response
                self.request_cache[cache_key] = response
                return response
            else:
                return "I couldn't process this request properly."
                
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Gemini API error: {str(e)}")
            
            # Check if it's a quota error (429)
            if "429" in str(e) or "quota" in error_str or "resource_exhausted" in error_str:
                # Extract retry delay if available
                import re as regex
                retry_match = regex.search(r'retry.*?(\d+(?:\.\d+)?)', error_str)
                if retry_match:
                    retry_seconds = float(retry_match.group(1))
                    self.quota_exceeded_until = time.time() + retry_seconds + 10
                    logger.warning(f"Quota exceeded. Waiting {retry_seconds} seconds")
                    return f"⏳ API quota exceeded. Waiting {int(retry_seconds)} seconds before retry..."
                else:
                    self.quota_exceeded_until = time.time() + 3600  # Wait 1 hour
                    logger.warning(f"Quota exceeded. Waiting 1 hour")
                    return "⏳ API quota limit reached. Please try again later (quota resets every 24 hours)."
            
            # Check for authentication errors (401)
            elif "401" in str(e) or "unauthenticated" in error_str or "invalid" in error_str:
                logger.error("API key authentication failed")
                return "❌ Authentication error with API key. Please check your configuration."
            
            # Check for model not found (404)
            elif "404" in str(e) or "not found" in error_str:
                logger.error("Model not found")
                return "❌ Model not available. The AI model may be temporarily unavailable."
            
            # Generic error
            else:
                return f"❌ Error: {str(e)[:100]}"
    
    def _generate_with_genai(self, prompt):
        """Generate content using google.genai API with fallback"""
        try:
            # Try gemini-2.5-pro first
            logger.info("Attempting with gemini-2.5-pro...")
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            elif response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    return ' '.join(part.text for part in parts if hasattr(part, 'text'))
            return None
            
        except Exception as e:
            error_str = str(e).lower()
            # If quota exceeded with pro, fall back to flash
            if "429" in str(e) or "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"gemini-2.5-pro quota exceeded, falling back to gemini-2.5-flash")
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    
                    if response and hasattr(response, 'text'):
                        return response.text
                    elif response and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            return ' '.join(part.text for part in parts if hasattr(part, 'text'))
                    return None
                except Exception as flash_error:
                    logger.error(f"gemini-2.5-flash also failed: {flash_error}")
                    raise flash_error
            else:
                logger.error(f"_generate_with_genai error: {e}")
                raise
    
    def clean_response(self, text):
        """Clean and format response with proper error handling"""
        if not text:
            return "❌ I couldn't generate a response."
        
        try:
            # Remove any problematic characters
            text = str(text).replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
            # Remove excessive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Add emoji prefix
            return "💡 " + text.strip()
        except Exception as e:
            logger.error(f"Error in clean_response: {str(e)}")
            return "❌ Error formatting response"

    def _is_programming_question(self, text):
        try:
            # Check if it's a programming question
            if chat_id and self._is_programming_question(query):
                # Store the question
                self.programming_questions[chat_id] = query
                # Create language selection buttons
                keyboard = [[
                    InlineKeyboardButton("🐍 Python", callback_data="lang_python"),
                    InlineKeyboardButton("☕ Java", callback_data="lang_java"),
                ], [
                    InlineKeyboardButton("⚡ C++", callback_data="lang_cpp"),
                    InlineKeyboardButton("💛 JavaScript", callback_data="lang_javascript")
                ]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                return ("Please select the programming language:", reply_markup)

        # Check for simple math
            if re.match(r'^[\d+\-*/().\s]+$', query):
                result = self.math.solve(query)
                if result is not None:
                    return f"🔢 Result: {result}"

        # Get Gemini response with error handling
            response = await self.get_gemini_response(query)
            if not response:
                return "❌ I couldn't generate a response. Please try again."
            
            return self.clean_response(response)

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return "❌ I encountered an error. Please try rephrasing your question."

    def _is_programming_question(self, text):
        keywords = [
            'program', 'code', 'function', 'algorithm',
            'write a', 'implement', 'create a program', 'Constraints:',
            'Input:', 'Output:', 'Example', 'return'
        ]
        return any(keyword.lower() in text.lower() for keyword in keywords)

bot = AIBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """
🌟 Welcome! I can help you with:

📊 Mathematics
🧮 Aptitude Problems
🔍 Pattern Recognition
📝 General Questions
💡 Technical Queries

Just ask me anything!
"""
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
I can help with:

• Math calculations
• Percentage problems
• Profit/Loss calculations
• Time and Distance
• Pattern Recognition
• Sequences and Series
• General knowledge
• Programming questions

Just type your question!
"""
    await update.message.reply_text(help_text)
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /stop command to temporarily stop the bot."""
    bot.is_active = False  # Just deactivate the bot
    await update.message.reply_text("🛑 Bot is now sleeping! Use /start to wake me up.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot.is_active = True  # Reactivate the bot
    welcome_text = """
🌟 Welcome! I can help you with:

📊 Mathematics
🧮 Aptitude Problems
🔍 Pattern Recognition
📝 General Questions
💡 Technical Queries

Just ask me anything!
"""
    await update.message.reply_text(welcome_text)

# Add new callback handler
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data.startswith("lang_"):
            language = query.data.split("_")[1]
            chat_id = query.message.chat_id
            
            if chat_id in bot.programming_questions:
                question = bot.programming_questions[chat_id]
                prompt = f"""Write a solution in {language} for the following problem:
                
{question}

Provide:
1. Problem analysis
2. Solution approach
3. Complete code with comments
4. Example usage
"""
                response = await bot.get_gemini_response(prompt)
                await query.edit_message_text(text=bot.clean_response(response))
                del bot.programming_questions[chat_id]  # Clean up
            else:
                await query.edit_message_text(text="❌ Question not found. Please ask again.")
                
    except Exception as e:
        logger.error(f"Error in button callback: {e}")
        await query.edit_message_text(text="❌ Error processing selection. Please try again.")

# Modify handle_message function
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        message_text = update.message.text

        if not await bot.should_respond(chat_id, message_text):
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        response = await bot.get_response(message_text, chat_id)
        
        # Check if response is a tuple (for programming questions)
        if isinstance(response, tuple):
            message_text, reply_markup = response
            await update.message.reply_text(message_text, reply_markup=reply_markup)
        else:
            if len(response) <= 4096:
                await update.message.reply_text(response)
            else:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
                
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error. Please try again.")

# Add this function before the main() function
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        
        # Check if bot should respond
        if not await bot.should_respond(chat_id, "image"):
            return
        
        # Get the photo with highest resolution
        photo = update.message.photo[-1]
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Download the photo
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        
        # Analyze the image
        await update.message.reply_text("🔍 Analyzing your image... This may take a moment.")
        response = await bot.analyze_image(photo_bytes)
        
        # Send response (handle long responses)
        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
                
    except Exception as e:
        logger.error(f"Error in handle_photo: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error processing your image. Please try again.")

# Update main() function to work with Railway
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
        handle_message
    ))
    # Add photo handler
    application.add_handler(MessageHandler(
        filters.PHOTO & filters.ChatType.GROUPS,
        handle_photo
    ))
    
    print("🤖 Bot is starting...")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
