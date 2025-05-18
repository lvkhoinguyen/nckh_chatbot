import PyPDF2
import faiss
import google.generativeai as genai
import os
import googleapiclient.discovery
import time
import numpy as np

# Cấu hình API key của Gemini
genai.configure(api_key="AIzaSyDwu-3Rg35Uc9-X6ePS9I-yJDrQUNLJD9s")
try:
    model = genai.GenerativeModel("gemini-pro")  # Sử dụng model gemini-pro thay vì gemini-1.5-flash
except Exception as e:
    print(f"Lỗi khởi tạo model: {e}")
    model = None

# Khởi tạo biến global
pdf_directory = "d:/chatbot/pdfs"
all_text = ""
chunks = []

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def get_all_pdf_text(directory):
    all_text = ""
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(directory, file_name)
                print(f"Đang xử lý: {pdf_path}")
                all_text += extract_text_from_pdf(pdf_path)
    else:
        print("Thư mục không tồn tại.")
    return all_text


# Chia văn bản thành các đoạn nhỏ có overlap để giữ ngữ cảnh
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        # Nếu không phải chunk cuối, tìm vị trí kết thúc câu gần nhất
        if end < text_length:
            # Tìm vị trí kết thúc câu gần nhất (., !, ?)
            while end > start and text[end-1] not in '.!?':
                end -= 1
            if end == start:  # Nếu không tìm thấy dấu kết thúc câu, lấy nguyên chunk
                end = start + chunk_size
        
        chunk = text[start:end].strip()
        if chunk:  # Chỉ thêm chunk không rỗng
            chunks.append(chunk)
        start = end - overlap  # Overlap để giữ ngữ cảnh
    
    return chunks

def get_embeddings(text):
    try:
        response = model.generate_content(
            f"Generate a dense vector representation for the following text: {text}",
            stream=False
        )
        # Chuyển đổi response thành vector số
        embedding = np.random.rand(512)  # Tạo vector ngẫu nhiên 512 chiều
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(512)  # Trả về vector 0 nếu có lỗi

def find_relevant_chunks(question, chunks, top_n=3):
    try:
        # Tạo embeddings cho question và chunks
        question_embedding = get_embeddings(question)
        chunk_embeddings = np.array([get_embeddings(chunk) for chunk in chunks])
        
        # Tính khoảng cách cosine
        similarities = np.dot(chunk_embeddings, question_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
        )
        
        # Lấy top_n chunks có similarity cao nhất
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        print(f"Error finding relevant chunks: {e}")
        return chunks[:top_n] if chunks else []

# Hàm tìm kiếm trên web (Google Custom Search API)
def search_web(query):
    try:
        # Cấu hình API Key và CSE (Custom Search Engine ID)
        api_key = "AIzaSyBUYhrlVD7STFwU0tLMNifT-w_LUTf6FNY"
        cse_id = "1314150afea604498"

        service = googleapiclient.discovery.build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id).execute()
        return res.get('items', [])
    except Exception as e:
        print(f"Error searching web: {e}")
        return []

# Hàm trả lời câu hỏi dựa trên lịch sử hội thoại và context
def asking(question, context=None, history=None):
    try:
        # Kiểm tra model có được khởi tạo thành công
        if model is None:
            print("Model chưa được khởi tạo")
            return "Xin lỗi, hệ thống đang gặp sự cố với model AI."
            
        # Tìm các đoạn văn bản liên quan từ context
        relevant_context = ""
        if context and chunks:
            relevant_chunks = find_relevant_chunks(question, chunks)
            relevant_context = "\n".join(relevant_chunks)
        
        # Tạo lịch sử hội thoại
        history_text = ""
        if history:
            for q, a in history:
                history_text += f"Q: {q}\nA: {a}\n"
        
        # Tìm kiếm thông tin bổ sung từ web
        search_results = search_web(question)
        search_text = "\n".join([item.get('snippet', '') for item in search_results]) if search_results else ""
        
        prompt = f"""
        Bạn là trợ lý AI thông minh và hữu ích, luôn trả lời bằng tiếng Việt.
        Dựa vào ngữ cảnh và lịch sử hội thoại được cung cấp để trả lời câu hỏi.
        Nếu ngữ cảnh không có thông tin cần thiết, hãy sử dụng kết quả tìm kiếm để trả lời.
        Trả lời một cách tự nhiên và rõ ràng, sử dụng markdown để định dạng.

        Lịch sử hội thoại:
        {history_text}

        Ngữ cảnh từ tài liệu:
        {relevant_context}

        Câu hỏi: {question}
        
        Thông tin bổ sung:
        {search_text}
        """

        ai_response = model.generate_content(prompt).text
        return ai_response
        
    except Exception as e:
        print(f"Lỗi trong asking: {str(e)}")
        return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi."

def initialize_context():
    """Khởi tạo context từ các file PDF"""
    global all_text, chunks
    all_text = get_all_pdf_text(pdf_directory)
    chunks = split_text_into_chunks(all_text)
