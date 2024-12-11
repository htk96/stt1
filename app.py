# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import asyncio
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
import signal
from groq import Groq
import numpy as np
import base64
from scipy.io import wavfile

# 환경 변수 로드
env_path = r"C:\Users\bmc\Desktop\홍태광\workspace\.env"
load_dotenv(env_path)

app = Flask(__name__)
app.config['DEEPGRAM_API_KEY'] = os.getenv("DEEPGRAM_API_KEY")
app.config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
app.config['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 변수로 종료 플래그 추가
is_running = True

# SafeMicrophone 클래스 구현
class SafeMicrophone(Microphone):
    def __init__(self, send_callback, loop=None):
        super().__init__(send_callback)
        self.loop = loop or asyncio.get_event_loop()
        
    async def _send_data(self, data):
        try:
            if asyncio.iscoroutinefunction(self.send_callback):
                await self.send_callback(data)
            else:
                await self.loop.run_in_executor(None, self.send_callback, data)
        except Exception as e:
            print(f"Error sending data: {e}")
            
    def send(self, data):
        asyncio.run_coroutine_threadsafe(self._send_data(data), self.loop)

class AudioProcessor:
    def __init__(self):
        self.buffer_size = 4096
        self.is_processing = False
        self.deepgram = None
        self.dg_connection = None
        self.is_connected = False
        self.loop = asyncio.new_event_loop()
        self.groq_client = Groq(api_key=app.config['GROQ_API_KEY'])
        self.audio_buffer = []
        self.last_process_time = datetime.now()
        
    def on_message(self, self_, result, **kwargs):
        try:
            # 트랜스크립션이 최종 결과인지 확인
            if not result.is_final:
                return
            
            transcript = result.channel.alternatives[0].transcript
            if len(transcript) == 0:
                return
                
            print(f"[Deepgram Nova-2] Recognized: {transcript}")
            
            # 클라이언트에 최종 트랜스크립션 전송
            socketio.emit('transcription_update', {
                'model': 'nova2',
                'text': transcript,
                'metrics': {
                    'latency': round(result.duration * 1000) if hasattr(result, 'duration') else 0,
                    'confidence': round(result.channel.alternatives[0].confidence * 100, 2) if hasattr(result.channel.alternatives[0], 'confidence') else 0
                }
            })
            
        except Exception as e:
            print(f"Error in message handler: {e}")

    async def init_deepgram(self):
        try:
            self.deepgram = DeepgramClient(api_key=app.config['DEEPGRAM_API_KEY'])
            
            self.dg_connection = self.deepgram.listen.live.v("1")
            
            # 이벤트 핸들러 등록
            self.dg_connection.on(LiveTranscriptionEvents.Open, lambda *args: print("Connection opened"))
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Close, lambda *args: print("Connection closed"))
            self.dg_connection.on(LiveTranscriptionEvents.Error, lambda *args: print("Error occurred"))
            
            options = LiveOptions(
                model="nova-2",
                language="ko",
                smart_format=True,
                interim_results=True,  # 임시 결과 비활성화 필요
                endpointing=200,
                punctuate=True,
                encoding='linear16',      # PCM 16-bit
                sample_rate=16000,       # 샘플 레이트 16kHz
                channels=1                # 모노
            )
            
            if not self.dg_connection.start(options):
                print("Failed to start Deepgram connection")
                return False
                
            print("Deepgram connection established successfully")
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Deepgram initialization error: {e}")
            return False

    async def cleanup(self):
        if self.dg_connection:
            self.dg_connection.finish()

    async def process_audio(self, audio_data):
        """오디오 데이터 처리"""
        try:
            # Deepgram Nova-2 처리
            if not self.is_connected:
                success = await self.init_deepgram()
                if not success:
                    return
            
            print(f"Processing audio data of size: {len(audio_data)} bytes")
            
            # Deepgram에 오디오 데이터 전송
            self.dg_connection.send(audio_data)
            print("Audio data sent to Deepgram successfully")
            
            # Groq Whisper 처리를 위한 버퍼링
            self.audio_buffer.append(audio_data)
            buffer_size_bytes = sum(len(chunk) for chunk in self.audio_buffer)
            
            # 버퍼가 일정 크기(약 2초)에 도달하면 처리
            if buffer_size_bytes >= 64000:  # 16000Hz * 2초 * 2바이트
                print("Processing buffered audio with Groq Whisper...")
                # 버퍼의 모든 오디오 데이터 결합
                combined_audio = np.concatenate([
                    np.frombuffer(chunk, dtype=np.int16) 
                    for chunk in self.audio_buffer
                ])
                
                # Groq 처리
                await self.process_audio_groq(combined_audio.tobytes())
                
                # 버퍼 초기화
                self.audio_buffer = []
                print("Groq processing completed and buffer cleared")
                
        except Exception as e:
            print(f"Error in process_audio: {e}")
            self.is_connected = False
            raise e

    async def process_audio_groq(self, audio_data):
        """Groq Whisper를 사용한 오디오 처리"""
        try:
            # PCM 데이터를 int16 배열로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # WAV 파일로 저장 (16kHz 샘플레이트)
            audio_file_path = "temp_audio.wav"
            print(f"Saving audio data to {audio_file_path}...")
            wavfile.write(audio_file_path, 16000, audio_array)
            
            # Groq API 호출 (음성 인식 요청)
            print("Sending audio to Groq API...")
            with open(audio_file_path, "rb") as file:
                response = self.groq_client.audio.transcriptions.create(
                    file=(audio_file_path, file.read()),
                    model="whisper-large-v3",
                    prompt="한국어 음성을 인식합니다.",
                    response_format="json",
                    temperature=0.0,
                    language="ko"
                )
                
                transcript = response.text
                print(f"[Groq Whisper] Recognized: {transcript}")
            
            # 임시 파일 삭제
            os.remove(audio_file_path)
            print("Temporary audio file removed")
            
            if transcript:
                # 클라이언트에 전송
                socketio.emit('transcription_update', {
                    'model': 'groq-whisper',
                    'text': transcript,
                    'metrics': {
                        'latency': round((datetime.now() - self.last_process_time).total_seconds() * 1000, 2),
                        'confidence': 0
                    }
                })
                
            self.last_process_time = datetime.now()
            
        except Exception as e:
            print(f"Groq Whisper processing error: {e}")
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

# STT/Translation 모델 구성
STT_MODELS = {
    'google': None,  # Google STT 클라이언트
    'whisper': None,  # OpenAI Whisper 클라이언트
    'nova2': None,  # Deepgram nova-2 클라이언트
    'whisper_large': None  # Deepgram whisper-large-v3 클라이언트
}

TRANSLATION_MODELS = {
    'llama_8b': None,  # Llama 3.1-8b 클라이언트
    'llama_70b': None,  # Llama 3.1-70b 클라이언트
    'gemma_7b': None  # Gemma 7b 클라이언트
}

# 오디오 처리를 위한 글로벌 큐
audio_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=8)

@app.route('/')
def index():
    return render_template('index.html')

audio_processor = AudioProcessor()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    if not audio_processor.is_processing:
        audio_processor.is_processing = True
        try:
            asyncio.run(audio_processor.process_audio(data))
        finally:
            audio_processor.is_processing = False

def signal_handler(signum, frame):
    global is_running
    print("\n종료 신호를 받았습니다. 정리 중...")
    is_running = False
    
    # ThreadPoolExecutor 종료
    executor.shutdown(wait=False)
    
    # 오디오 프로세서 정리
    asyncio.run(audio_processor.cleanup())
    
    # 소켓 연결 종료
    socketio.stop()

if __name__ == '__main__':
    try:
        # AudioProcessor 초기화
        audio_processor = AudioProcessor()
        # 초기 Deepgram 연결 설정
        audio_processor.loop.run_until_complete(audio_processor.init_deepgram())
        
        socketio.run(app, debug=True)
    finally:
        audio_processor.loop.close()
        print("프로그램을 종료합니다.")