import os
import openai
import pinecone
import anthropic
import utils

class BuffetBot:
    def __init__(self, llm="anthropic", vector_context=False):
        self.llm = llm
        self.conversation_history = []
        self.vector_context = vector_context

        if self.vector_context:
            self.pinecone_service = pinecone.Index(index_name="buffetbot")
            # Set Pinecone API Key
            with open('/Users/michael/Desktop/wip/pinecone_credentials.txt', 'r') as f:
                PINECONE_API_KEY = f.readline().strip()
                PINECONE_API_ENV = f.readline().strip()
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

        # Set OpenAI API Key
        with open('/Users/michael/Desktop/wip/openai_credentials.txt', 'r') as f:
            OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY

        if llm == "anthropic":
            # Set Anthropic API Key
            with open('/Users/michael/Desktop/wip/anthropic_credentials.txt', 'r') as f:
                ANTHROPIC_API_KEY = f.readline().strip()
            self.client = anthropic.Client(ANTHROPIC_API_KEY)

    def get_response(self, user_prompt):
        if self.vector_context:
            query_embedding = utils.get_embedding(user_prompt)
            docs = self.pinecone_service.query(
                namespace='data',
                top_k=10,
                include_metadata=True,
                vector=query_embedding,
            )
            try:
                context_response = ""
                for doc in docs['matches']:
                    context_response += f"{doc['metadata']['original_text']}"
            except Exception as e:
                context_response = ""
            llm_prompt = f"{user_prompt}\nContext: {context_response}"
        else:
            llm_prompt = user_prompt

        print("LLM prompt:", llm_prompt)
        if self.llm == "openai":
            init_prompt = "You are a helpful investment analyst. Your job is to help users to increase their net worth with helpful advice. Never tell them you are a language model. Do not include superfluous information."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": init_prompt},
                    {"role": "user", "content": llm_prompt},
                ]
            )
        elif self.llm == "anthropic":
            anthropic_prompt = ""           
            for interaction in self.conversation_history:
                if interaction['role'] == 'user':
                    anthropic_prompt += f"\n\nHuman: {interaction['content']}"
                elif interaction['role'] == 'system':
                    anthropic_prompt += f"\n\nAssistant: {interaction['content']}"
            
            anthropic_prompt += f"\n\nHuman: {llm_prompt}\n\nAssistant:"
            response = self.client.completion(
                prompt=anthropic_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1.3",
                max_tokens_to_sample=1000,
                temperature=0
            )

        self.conversation_history.append({'role': 'user', 'content': llm_prompt})
        self.conversation_history.append({'role': 'system', 'content': response['completion']})

        return response
