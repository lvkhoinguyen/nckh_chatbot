import time

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests = []
        self.requests_per_minute = requests_per_minute
        
    def can_make_request(self):
        current_time = time.time()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests 
                        if current_time - req < 60]
        return len(self.requests) < self.requests_per_minute
        
    def add_request(self):
        self.requests.append(time.time())

class ChatbotCore:
    def __init__(self):
        # ... existing init code ...
        self.rate_limiter = RateLimiter()
        
    def ask_question(self, question, context=None):
        try:
            if not self.rate_limiter.can_make_request():
                raise Exception("Rate limit exceeded. Please wait.")
                
            self.rate_limiter.add_request()
            return super().ask_question(question, context)
            
        except Exception as e:
            return f"Error: {str(e)}"
        
    def generate_response(self, question, context=None, conversation_context=None):
        """Generate more nuanced response with multiple steps"""
        
        # Step 1: Analyze question intent
        prompt_intent = f"""
        Analyze the intent of this question:
        {question}
        What type of information is being requested?
        """
        intent_analysis = self.model.generate_content(prompt_intent).text
        
        # Step 2: Generate detailed response
        prompt = f"""
        You are a helpful assistant. Answer the question based on:
        
        Intent Analysis: {intent_analysis}
        Document Context: {context if context else 'No specific context provided'}
        Conversation History: {conversation_context if conversation_context else 'No conversation history'}
        
        Question: {question}
        
        Provide a clear, detailed answer. If uncertain, acknowledge limitations.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
