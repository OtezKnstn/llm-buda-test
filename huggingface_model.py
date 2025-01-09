import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_text_gen(batch_size=1):
    # Загрузка модели и токенизатора
    model_name = "decapoda-research/llama-7b-hf"  # Укажите нужную модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Входной текст для генерации
    input_text = "Once upon a time in a distant galaxy,"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size)
    # Генерация текста
    output_ids = text_generator(
        input_ids,
        max_length=50,  # Максимальная длина генерируемого текста
        num_return_sequences=1,  # Количество сгенерированных вариантов
        temperature=0.7,  # Контроль креативности (меньше = более детерминированный результат)
        top_p=0.9,  # Контроль разнообразия (nucleus sampling)
        do_sample=True,  # Включить семплирование
    )

    # Декодирование и вывод результата
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Сгенерированный текст:", output_text)

if __name__ == "__main__":
    while True:
        run_text_gen()