class ConversationManager:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        
    def add_interaction(self, question, answer):
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    def get_context(self, window_size=3):
        """Get recent conversation context"""
        recent = self.conversation_history[-window_size:]
        context = ""
        for interaction in recent:
            context += f"Q: {interaction['question']}\nA: {interaction['answer']}\n\n"
        return context