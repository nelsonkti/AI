import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    # 加载微调后的模型和tokenizer
    model_name = './fine_tuned_model'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 设置模型为评估模式
    model.eval()

    # 定义聊天函数
    def chat_with_bot(prompt, max_length=100):
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_p=0.95,
                top_k=50,
                temperature=0.7
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # 开始聊天
    print("Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_bot(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
