import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_text_gen(batch_size=1):
    # Загрузка модели и токенизатора
    model_name = "gai-labs/strela"  # Используем открытую модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Настройка компилятора Buda
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b  # Используем формат FP16

    # Компиляция модели
    model = pybuda.compile_model(model, compiler_cfg)

    # Создание пайплайна для генерации текста
    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size)

    # Входной текст для генерации
    input_text = "Once upon a time in a distant galaxy,"
    result = text_generator(input_text, max_length=50, temperature=0.7, top_p=0.9, do_sample=True)

    # Вывод результата
    print("Сгенерированный текст:", result[0]['generated_text'])

if __name__ == "__main__":
    run_text_gen()
