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
from google.cloud import speech_v1
from google.cloud.speech_v1 import types

# 환경 변수 로드
env_path = r"C:\Users\bmc\Desktop\홍태광\workspace\.env"
load_dotenv(env_path)

# Google Cloud 인증 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app = Flask(__name__)
app.config['DEEPGRAM_API_KEY'] = os.getenv("DEEPGRAM_API_KEY")
app.config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
app.config['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 변수로 종료 플래그 추가
is_running = True

# 로그 색상 정의 추가
class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_with_color(model, message, color):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{color}[{timestamp}] [{model}] {message}{LogColors.ENDC}")

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
        
        # Google STT 클라이언트 초기화 추가
        try:
            self.google_client = speech_v1.SpeechClient()
            log_with_color("Google STT", "클라이언트 초기화 성공", LogColors.GREEN)
        except Exception as e:
            log_with_color("Google STT", f"클라이언트 초기화 실패: {e}", LogColors.RED)
            
    def on_message(self, self_, result, **kwargs):
        try:
            if not result.is_final:
                log_with_color("Deepgram", "중간 결과 수신", LogColors.BLUE)
                return
            
            transcript = result.channel.alternatives[0].transcript
            if len(transcript) == 0:
                log_with_color("Deepgram", "빈 트랜스크립트 수신", LogColors.YELLOW)
                return
            
            confidence = result.channel.alternatives[0].confidence if hasattr(result.channel.alternatives[0], 'confidence') else 0
            duration = result.duration if hasattr(result, 'duration') else 0
            
            log_with_color("Deepgram Nova-2", 
                f"인식 결과:\n"
                f"텍스트: {transcript}\n"
                f"신뢰도: {round(confidence * 100, 2)}%\n"
                f"처리시간: {round(duration * 1000)}ms", 
                LogColors.GREEN)
            
            socketio.emit('transcription_update', {
                'model': 'nova2',
                'text': transcript,
                'metrics': {
                    'latency': round(duration * 1000),
                    'confidence': round(confidence * 100, 2)
                }
            })
            
        except Exception as e:
            log_with_color("Deepgram Nova-2", f"결과 처리 중 오류 발생: {e}", LogColors.RED)

    async def init_deepgram(self):
        try:
            self.deepgram = DeepgramClient(api_key=app.config['DEEPGRAM_API_KEY'])
            log_with_color("Deepgram", "클라이언트 초기화 성공", LogColors.GREEN)
            
            self.dg_connection = self.deepgram.listen.live.v("1")
            log_with_color("Deepgram", "연결 객체 생성 성공", LogColors.GREEN)
            
            # 이벤트 핸들러 등록
            self.dg_connection.on(LiveTranscriptionEvents.Open, 
                lambda *args: log_with_color("Deepgram", "연결이 열렸습니다", LogColors.GREEN))
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Close, 
                lambda *args: log_with_color("Deepgram", "연결이 닫혔습니다", LogColors.YELLOW))
            self.dg_connection.on(LiveTranscriptionEvents.Error, 
                lambda *args: log_with_color("Deepgram", f"에러 발생: {args}", LogColors.RED))
            
            options = LiveOptions(
                model="nova-2",
                language="ko",
                smart_format=True,
                interim_results=True,
                endpointing=200,
                punctuate=True,
                encoding='linear16',
                sample_rate=16000,
                channels=1
            )
            
            log_with_color("Deepgram", f"설정된 옵션: {options}", LogColors.BLUE)
            
            if not self.dg_connection.start(options):
                log_with_color("Deepgram", "연결 시작 실패", LogColors.RED)
                return False
            
            log_with_color("Deepgram", "연결이 성공적으로 설정되었습니다", LogColors.GREEN)
            self.is_connected = True
            return True
            
        except Exception as e:
            log_with_color("Deepgram", f"초기화 중 오류 발생: {e}", LogColors.RED)
            return False

    async def cleanup(self):
        if self.dg_connection:
            log_with_color("Deepgram", "연결 종료 시작", LogColors.YELLOW)
            self.dg_connection.finish()
            log_with_color("Deepgram", "연결이 정상적으로 종료되었습니다", LogColors.GREEN)

    async def process_audio(self, audio_data):
        """오디오 데이터 처리"""
        try:
            # Deepgram Nova-2 처리
            if not self.is_connected:
                log_with_color("Deepgram", "연결이 없습니다. 재연결 시도...", LogColors.YELLOW)
                success = await self.init_deepgram()
                if not success:
                    log_with_color("Deepgram", "재연결 실패", LogColors.RED)
                    return
            
            log_with_color("Deepgram", f"오디오 데이터 처리 시작 (크기: {len(audio_data)} bytes)", LogColors.BLUE)
            
            # Deepgram에 오디오 데이터 전송
            try:
                self.dg_connection.send(audio_data)
                log_with_color("Deepgram", "오디오 데이터 전송 성공", LogColors.GREEN)
            except Exception as e:
                log_with_color("Deepgram", f"오디오 데이터 전송 실패: {e}", LogColors.RED)
                self.is_connected = False
                raise e
            
            # Google STT 처리 추가
            try:
                await self.process_audio_google(audio_data)
            except Exception as e:
                log_with_color("Google STT", f"처리 중 오류 발생: {e}", LogColors.RED)
            
            # Groq Whisper 처리를 위한 버퍼링
            self.audio_buffer.append(audio_data)
            buffer_size_bytes = sum(len(chunk) for chunk in self.audio_buffer)
            
            if buffer_size_bytes >= 64000:
                log_with_color("Audio Processing", "Processing buffered audio...", LogColors.BLUE)
                combined_audio = np.concatenate([
                    np.frombuffer(chunk, dtype=np.int16) 
                    for chunk in self.audio_buffer
                ])
                
                await self.process_audio_groq(combined_audio.tobytes())
                self.audio_buffer = []
                
        except Exception as e:
            log_with_color("Deepgram", f"오디오 처리 중 오류 발생: {e}", LogColors.RED)
            self.is_connected = False
            raise e

    async def process_audio_google(self, audio_data):
        """Google STT medical_dictation 모델을 사용한 실시간 음성 인식"""
        try:
            start_time = datetime.now()
            
            # 설정
            config = types.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR",
                model="default",
                enable_automatic_punctuation=True,
                use_enhanced=True
            )
            
            streaming_config = types.StreamingRecognitionConfig(
                config=config,
                interim_results=True
            )
            
            log_with_color("Google STT", f"오디오 데이터 크기: {len(audio_data)} bytes", LogColors.YELLOW)
            
            # 스트리밍 요청 생성
            def request_generator():
                # 첫 번째 요청은 설정만 포함
                yield speech_v1.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                
                # 오디오 데이터를 청크로 분할하여 전송
                audio_generator = np.frombuffer(audio_data, dtype=np.int16)
                chunk_size = 1024
                
                for i in range(0, len(audio_generator), chunk_size):
                    chunk = audio_generator[i:i + chunk_size]
                    yield speech_v1.StreamingRecognizeRequest(
                        audio_content=chunk.tobytes()
                    )
            
            log_with_color("Google STT", "스트리밍 인식 시작", LogColors.YELLOW)
            
            # 스트리밍 인식 수행
            requests = request_generator()
            responses = self.google_client.streaming_recognize(requests)
            
            for response in responses:
                if not response.results:
                    continue
                    
                result = response.results[0]
                if not result.alternatives:
                    continue
                    
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                is_final = result.is_final
                
                if is_final:
                    log_with_color("Google STT", f"Recognized: {transcript}", LogColors.YELLOW)
                    
                    socketio.emit('transcription_update', {
                        'model': 'google-medical',
                        'text': transcript,
                        'metrics': {
                            'latency': round((datetime.now() - start_time).total_seconds() * 1000, 2),
                            'confidence': round(confidence * 100, 2) if confidence else 0
                        }
                    })
                    
        except Exception as e:
            log_with_color("Google STT", f"Error: {str(e)}", LogColors.RED)
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
                # response = self.groq_client.audio.translations.create(
                response = self.groq_client.audio.transcriptions.create(
                    file=(audio_file_path, file.read()),
                    model="whisper-large-v3",
                    prompt="한국어 음성을 인식합니다.",
                    response_format="json",
                    temperature=0.0,
                    language="ko"
                )
                
                transcript = response.text
                log_with_color("Groq Whisper", f"Recognized: {transcript}", LogColors.BLUE)
            
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
            log_with_color("Groq Whisper", f"Error: {e}", LogColors.RED)
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