from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

yt = YouTubeTranscriptApi()

def get_timestamped_docs(video_id: str) -> list[Document]:
    try:
        transcript = yt.fetch(video_id)
    
        chunks, current_words, current_start, current_end = [], [], None, None
    
        for entry in transcript:
            words = entry.text.split()
            if current_start is None:
                current_start = entry.start
    
            current_words.extend(words)
            current_end = entry.start+ entry.duration
    
            if len(current_words) >= CHUNK_SIZE:
                chunks.append({
                    'text': ' '.join(current_words),
                    'start': current_start,
                    'end': current_end,
                })
                current_words = current_words[-CHUNK_OVERLAP:]
                current_start = current_end
    
        if current_words:
            chunks.append({'text': ' '.join(current_words), 'start': current_start, 'end': current_end})
    
        docs = [
            Document(
                page_content=chunk['text'],
                metadata={
                    'video_id': video_id,
                    'start_time': chunk['start'],
                    'end_time': chunk['end'],
                    'yt_url': f'https://youtube.com/watch?v={video_id}&t={int(chunk['start'])}',
                }
            )for chunk in chunks]
        return docs
    except TranscriptsDisabled:
        raise ValueError(f'Transcripts are disabled for video: {video_id}')
    except NoTranscriptFound:
        raise ValueError(f'No transcript found for video: {video_id}')
    except Exception as e:
        raise RuntimeError(f'Failed to fetch transcript: {e}')
    
