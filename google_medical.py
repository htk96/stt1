import os
from google.cloud import speech_v1
from google.cloud import translate_v2
from google.oauth2 import service_account
import pyaudio
import wave
import threading
import queue
import time
from dotenv import load_dotenv

# 현재 파일의 디렉토리 경로를 가져옴
current_dir = os.path.dirname(os.path.abspath(__file__))
# .env 파일 경로 설정
env_path = os.path.join(current_dir, '.env')
# .env 파일 로드
load_dotenv(env_path)

class MedicalConferenceSTT:
    def __init__(self):
        try:
            # 서비스 계정 키 파일 경로 가져오기
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in .env file")
            
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
                
            # 서비스 계정 인증 정보 생성
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            
            # 인증된 클라이언트 생성
            self.client = speech_v1.SpeechClient(credentials=credentials)
            self.translate_client = translate_v2.Client(credentials=credentials)
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            print(f"현재 디렉토리: {current_dir}")
            print(f"환경 변수 파일 경로: {env_path}")
            print(f"자격 증명 파일 경로: {credentials_path if 'credentials_path' in locals() else 'Not set'}")
            raise
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        
        # STT 설정
        self.config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            model="medical_dictation",
            enable_automatic_punctuation=True,
            use_enhanced=True,
            metadata=speech_v1.RecognitionMetadata(
                interaction_type=speech_v1.RecognitionMetadata.InteractionType.DISCUSSION,
                industry_naics_code_of_audio=621111,
                microphone_distance=speech_v1.RecognitionMetadata.MicrophoneDistance.NEARFIELD,
                original_media_type=speech_v1.RecognitionMetadata.OriginalMediaType.AUDIO,
                recording_device_type=speech_v1.RecognitionMetadata.RecordingDeviceType.PC,
            )
        )

    def start_recording(self):
        """마이크 녹음 시작"""
        self.is_recording = True
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("녹음 시작...")
        
        # 녹음 스레드 시작
        threading.Thread(target=self._record_audio).start()
        # STT 처리 스레드 시작
        threading.Thread(target=self._process_audio).start()

    def _record_audio(self):
        """오디오 녹음 및 큐에 저장"""
        while self.is_recording:
            data = self.stream.read(self.CHUNK)
            self.audio_queue.put(data)

    def _process_audio(self):
        """음성 인식 및 번역 처리"""
        streaming_config = speech_v1.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True  # 중간 결과 활성화
        )

        def generate_requests():
            while self.is_recording:
                if not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    yield speech_v1.StreamingRecognizeRequest(audio_content=data)

        try:
            requests = generate_requests()
            responses = self.client.streaming_recognize(
                config=streaming_config,
                requests=requests
            )

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                if result.is_final:
                    print(f"\n원본 (영어): {transcript}")
                    
                    # 한국어로 번역
                    translation = self.translate_client.translate(
                        transcript,
                        target_language='ko',
                        source_language='en'
                    )
                    
                    print(f"번역 (한국어): {translation['translatedText']}")
                    print("-" * 50)
                else:
                    print(f"\r임시 인식 결과: {transcript}", end="")

        except Exception as e:
            print(f"Error during recognition: {e}")

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("\n녹음 종료")

def main():
    stt = MedicalConferenceSTT()
    
    try:
        stt.start_recording()
        # 테스트를 위해 30초 동안 실행
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    finally:
        stt.stop_recording()

if __name__ == "__main__":
    main()