import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import av
import speech_recognition as sr
import pyttsx3
import numpy as np
import threading
import queue
import time
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# Configure page
st.set_page_config(
    page_title="🚖 Voice Taxi Dispatcher (WebRTC)", 
    page_icon="🚖",
    layout="wide"
)

# Initialize OpenAI client
api_key = st.secrets("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found! Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

class VoiceProcessor(AudioProcessorBase):
    """Audio processor for WebRTC voice processing"""
    
    def __init__(self):
        self.audio_buffer = []
        self.recognizer = sr.Recognizer()
        # Adjust recognition settings for better performance
        self.recognizer.energy_threshold = 100  # Lower for more sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.3
        self.is_processing = False
        self.sample_rate = 16000
        self.last_process_time = 0
        
    def recv_queued(self, frames):
        """Process incoming audio frames"""
        if not frames:
            return frames
            
        for frame in frames:
            # Convert audio to numpy array
            audio_array = frame.to_ndarray()
            
            # Handle stereo to mono conversion
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            elif len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Calculate volume for feedback
            if len(audio_array) > 0:
                rms = np.sqrt(np.mean(audio_array**2))
                volume_level = min(100, int(rms * 10000))
                
                # Send volume to session state via queue
                if 'volume_queue' not in st.session_state:
                    st.session_state.volume_queue = queue.Queue(maxsize=10)
                
                try:
                    if not st.session_state.volume_queue.full():
                        st.session_state.volume_queue.put_nowait(volume_level)
                except:
                    pass
            
            # Add to buffer
            self.audio_buffer.extend(audio_array)
            
            # Process when we have enough audio (2 seconds worth)
            buffer_length = len(self.audio_buffer)
            if buffer_length >= self.sample_rate * 2 and not self.is_processing:
                current_time = time.time()
                # Prevent too frequent processing
                if current_time - self.last_process_time > 1.0:
                    self.last_process_time = current_time
                    # Process in separate thread
                    threading.Thread(target=self._process_audio, daemon=True).start()
        
        return frames
    
    def _process_audio(self):
        """Process audio buffer for speech recognition"""
        if self.is_processing or len(self.audio_buffer) < self.sample_rate:
            return
            
        self.is_processing = True
        
        try:
            # Send processing status
            if 'status_queue' not in st.session_state:
                st.session_state.status_queue = queue.Queue(maxsize=10)
            
            try:
                if not st.session_state.status_queue.full():
                    st.session_state.status_queue.put_nowait("processing")
            except:
                pass
            
            # Get audio data (last 3 seconds)
            audio_length = min(len(self.audio_buffer), self.sample_rate * 3)
            audio_data = np.array(self.audio_buffer[-audio_length:], dtype=np.float32)
            
            # Clear old buffer but keep some overlap
            if len(self.audio_buffer) > self.sample_rate * 2:
                self.audio_buffer = self.audio_buffer[-self.sample_rate:]
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Create AudioData for speech recognition
            audio_source = sr.AudioData(audio_bytes, self.sample_rate, 2)
            
            # Try speech recognition
            try:
                # First try English (usually more reliable)
                text = self.recognizer.recognize_google(audio_source, language='en-US')
                if text and text.strip():
                    self._send_text(text.strip())
            except sr.UnknownValueError:
                # Try Serbian if English fails
                try:
                    text = self.recognizer.recognize_google(audio_source, language='sr-RS')
                    if text and text.strip():
                        self._send_text(text.strip())
                except sr.UnknownValueError:
                    pass  # No speech detected
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                
        except Exception as e:
            print(f"Audio processing error: {e}")
        finally:
            self.is_processing = False
            # Send idle status
            try:
                if 'status_queue' in st.session_state and not st.session_state.status_queue.full():
                    st.session_state.status_queue.put_nowait("idle")
            except:
                pass
    
    def _send_text(self, text):
        """Send detected text to main thread"""
        try:
            if 'text_queue' not in st.session_state:
                st.session_state.text_queue = queue.Queue(maxsize=10)
            
            if not st.session_state.text_queue.full():
                st.session_state.text_queue.put_nowait(text)
                print(f"Detected speech: {text}")  # Debug log
        except Exception as e:
            print(f"Error sending text: {e}")

def get_ai_response(conversation_history: List[tuple], user_message: str) -> str:
    """Get AI response using OpenAI API"""
    try:
        messages = [
            {
                "role": "system", 
                "content": """Ti si srpski taksi dispečer. Govori isključivo srpski (ekavica).
                
                Tvoj zadatak:
                1) Pozdravi korisnika profesionalno
                2) Pitaj za tačnu adresu preuzimanja
                3) Pitaj za broj putnika  
                4) Pitaj za odredište
                5) Potvrdi sve podatke
                6) Završi sa 'Vaš taksi stiže za 5-10 minuta. Hvala na pozivu i prijatan dan!'
                
                Budi kratak, profesionalan i ljubazan. Koristi standardne srpske fraze za taksi službu.
                Odgovori u maksimalno 2 rečenice."""
            }
        ]
        
        # Add conversation history
        for speaker, message in conversation_history:
            if speaker == "Bot":
                messages.append({"role": "assistant", "content": message})
            else:
                messages.append({"role": "user", "content": message})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return "Izvinjavam se, imamo tehnički problem. Molim pokušajte ponovo."

def extract_booking_data(conversation_history: List[tuple]) -> Dict:
    """Extract booking data from conversation"""
    try:
        conversation_text = "\n".join([f"{speaker}: {message}" for speaker, message in conversation_history])
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Izvuci podatke o rezervaciji taksija iz razgovora. 
                    Vrati JSON sa sledećim poljima:
                    - pickup_address: adresa preuzimanja
                    - destination: odredište  
                    - passengers: broj putnika (broj)
                    - status: "pending" ili "confirmed"
                    
                    Ako neki podatak nije spomenut, ostavi prazan string ili null."""
                },
                {
                    "role": "user", 
                    "content": f"Razgovor:\n{conversation_text}"
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        data = json.loads(response.choices[0].message.content)
        return data
        
    except Exception as e:
        st.error(f"Data extraction error: {e}")
        return {}

def speak_text(text: str):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        
        # Set up voice
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def main():
    st.title("🚖 Voice Taxi Dispatcher (WebRTC)")
    st.markdown("**Real-time voice conversation** - Speak directly into your microphone!")
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'booking_data' not in st.session_state:
        st.session_state.booking_data = {}
    if 'call_active' not in st.session_state:
        st.session_state.call_active = False
    if 'volume_level' not in st.session_state:
        st.session_state.volume_level = 0
    if 'last_detected_text' not in st.session_state:
        st.session_state.last_detected_text = ""
    if 'is_processing_speech' not in st.session_state:
        st.session_state.is_processing_speech = False
    
    # Initialize queues
    if 'volume_queue' not in st.session_state:
        st.session_state.volume_queue = queue.Queue(maxsize=10)
    if 'status_queue' not in st.session_state:
        st.session_state.status_queue = queue.Queue(maxsize=10)
    if 'text_queue' not in st.session_state:
        st.session_state.text_queue = queue.Queue(maxsize=10)
    
    # Process queued messages
    # Update volume level
    try:
        while not st.session_state.volume_queue.empty():
            st.session_state.volume_level = st.session_state.volume_queue.get_nowait()
    except queue.Empty:
        pass
    
    # Update processing status
    try:
        while not st.session_state.status_queue.empty():
            status = st.session_state.status_queue.get_nowait()
            st.session_state.is_processing_speech = (status == "processing")
    except queue.Empty:
        pass
    
    # Process detected text
    try:
        while not st.session_state.text_queue.empty():
            detected_text = st.session_state.text_queue.get_nowait()
            st.session_state.last_detected_text = detected_text
            
            # Process the speech immediately
            if st.session_state.call_active:
                st.balloons()
                
                # Add user message to conversation
                st.session_state.conversation.append(("User", detected_text))
                
                # Get AI response
                with st.spinner("🤖 Dispečer razmišlja..."):
                    ai_response = get_ai_response(st.session_state.conversation[:-1], detected_text)
                    st.session_state.conversation.append(("Bot", ai_response))
                    
                    # Extract booking data
                    booking_data = extract_booking_data(st.session_state.conversation)
                    if booking_data:
                        st.session_state.booking_data.update(booking_data)
                    
                    # Speak the AI response
                    threading.Thread(target=speak_text, args=(ai_response,), daemon=True).start()
                
                st.rerun()
    except queue.Empty:
        pass
    
    # WebRTC Configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Main controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📞 Start Call", disabled=st.session_state.call_active):
            st.session_state.call_active = True
            initial_greeting = "Dobro jutro! Taksi služba 'Brzi Prevoz', kako mogu da vam pomognem?"
            st.session_state.conversation = [("Bot", initial_greeting)]
            
            # Speak the greeting
            threading.Thread(target=speak_text, args=(initial_greeting,), daemon=True).start()
            st.rerun()
    
    with col2:
        if st.button("📞 End Call", disabled=not st.session_state.call_active):
            st.session_state.call_active = False
            st.session_state.conversation = []
            st.session_state.booking_data = {}
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.conversation = []
            st.session_state.booking_data = {}
            st.session_state.call_active = False
            st.session_state.last_detected_text = ""
            st.session_state.volume_level = 0
            st.rerun()
    
    # WebRTC Audio Streamer
    if st.session_state.call_active:
        st.markdown("### 🎤 Voice Input")
        
        # Voice status indicators
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            volume = st.session_state.volume_level
            if volume > 20:
                st.success(f"🔊 Volume: {volume}% - Good!")
            elif volume > 5:
                st.warning(f"🔉 Volume: {volume}% - Speak louder")
            else:
                st.info(f"🔈 Volume: {volume}% - Waiting...")
        
        with status_col2:
            if st.session_state.is_processing_speech:
                st.info("🔄 Processing speech...")
            else:
                st.success("👂 Listening...")
        
        with status_col3:
            if st.session_state.last_detected_text:
                st.success(f"🎤 Heard: '{st.session_state.last_detected_text[:30]}...'")
            else:
                st.info("🎤 No speech detected yet")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="voice-taxi",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=VoiceProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "sampleRate": 16000,
                    "channelCount": 1,
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True
                }
            },
            async_processing=True
        )
        
        # Show connection status
        if webrtc_ctx.state.playing:
            st.success("🎤 Microphone connected and recording!")
        else:
            st.error("🎤 Please allow microphone access and click 'START' above")
        
        # Manual text input for testing
        st.markdown("### 📝 Manual Input (for testing)")
        manual_input = st.text_input("Type your message here:", key="manual_input")
        if st.button("Send Manual Message") and manual_input:
            st.session_state.text_queue.put(manual_input)
            st.rerun()
        
        # Instructions
        st.info("""
        💡 **Testing Tips:**
        - Speak clearly and loudly into your microphone
        - Watch the volume indicator - should be 20%+ for best results
        - Try saying: "Hello, I need a taxi" or "Dobar dan, trebam taksi"
        - Use manual input above to test without voice
        """)
    
    else:
        st.info("Click 'Start Call' to begin voice conversation")
    
    # Display conversation
    if st.session_state.conversation:
        st.subheader("📞 Live Conversation")
        
        for speaker, message in st.session_state.conversation:
            if speaker == "Bot":
                with st.chat_message("assistant", avatar="🚖"):
                    st.write(f"**Dispatcher**: {message}")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.write(f"**You**: {message}")
    
    # Show booking data
    if st.session_state.booking_data:
        st.subheader("📋 Booking Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if 'pickup_address' in st.session_state.booking_data and st.session_state.booking_data['pickup_address']:
                st.metric("📍 Pickup", st.session_state.booking_data['pickup_address'])
            if 'passengers' in st.session_state.booking_data and st.session_state.booking_data['passengers']:
                st.metric("👥 Passengers", st.session_state.booking_data['passengers'])
        
        with info_col2:
            if 'destination' in st.session_state.booking_data and st.session_state.booking_data['destination']:
                st.metric("🎯 Destination", st.session_state.booking_data['destination'])
            if 'status' in st.session_state.booking_data and st.session_state.booking_data['status']:
                st.metric("📋 Status", st.session_state.booking_data['status'])
    
    # Sidebar with debug info
    with st.sidebar:
        st.markdown("### 🔧 Debug Information")
        st.write(f"**Call Active:** {st.session_state.call_active}")
        st.write(f"**Volume Level:** {st.session_state.volume_level}%")
        st.write(f"**Processing:** {st.session_state.is_processing_speech}")
        st.write(f"**Last Detected:** {st.session_state.last_detected_text}")
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Click **Start Call**
        2. Allow microphone access
        3. Speak clearly into microphone
        4. Watch volume indicator (needs 20%+)
        5. Use manual input to test
        """)

if __name__ == "__main__":

    main()
