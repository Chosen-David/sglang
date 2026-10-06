import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparse_attn.arguments import parse_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "true"
model_name = "hf_models/downloads/Qwen/Qwen3-8B" # meta-llama/Meta-Llama-3.1-8B-Instruct

print("model_name: {}".format(model_name))

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_new_tokens = 128

prompts = [
    [
        {
            "role": "user", "content": (
                "Please help me implement the python code that print \"Hello World!\"."
            )
        },
    ],
]
# prompts = [
#     [
#         {
#                 "role": "user", "content": (
# """
# Below is a record of lines I want you to remember. Each line begins with 'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the line as a numerical value. For each line index, memorize its corresponding <REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the corresponding <REGISTER_CONTENT> of a certain line index. Now the record start:
# line draw-alpenhorn: REGISTER_CONTENT is <36100>
# line vintage-peaceful: REGISTER_CONTENT is <46133>
# line fee-trigger: REGISTER_CONTENT is <23897>
# line acquisition-elf: REGISTER_CONTENT is <33081>
# line quarrelsome-ratepayer: REGISTER_CONTENT is <40346>
# line cravat-strive: REGISTER_CONTENT is <17010>
# line driving-specify: REGISTER_CONTENT is <15176>
# line spend-yell: REGISTER_CONTENT is <23911>
# line distribute-venture: REGISTER_CONTENT is <25499>
# line counter-force-pathogenesis: REGISTER_CONTENT is <37399>
# line carotene-visitor: REGISTER_CONTENT is <16831>
# line halting-speed: REGISTER_CONTENT is <2900>
# line statement-surface: REGISTER_CONTENT is <34357>
# line nice-arm: REGISTER_CONTENT is <26918>
# line vitamin-sake: REGISTER_CONTENT is <4514>
# line collection-moustache: REGISTER_CONTENT is <5516>
# line fertile-bed: REGISTER_CONTENT is <2849>
# line stir-fry-swordfight: REGISTER_CONTENT is <14442>
# line prove-suppose: REGISTER_CONTENT is <8104>
# line abdomen-select: REGISTER_CONTENT is <10886>
# line envelope-bra: REGISTER_CONTENT is <2994>
# line family-wad: REGISTER_CONTENT is <6276>
# line dress-lion: REGISTER_CONTENT is <12022>
# line crown-apple: REGISTER_CONTENT is <5609>
# line woman-union: REGISTER_CONTENT is <23928>
# line iron-fatigues: REGISTER_CONTENT is <48812>
# line parole-phrasing: REGISTER_CONTENT is <31590>
# line ripple-hydrolysis: REGISTER_CONTENT is <16870>
# line pathology-administrator: REGISTER_CONTENT is <1955>
# line supplier-survival: REGISTER_CONTENT is <1969>
# line velocity-verve: REGISTER_CONTENT is <28023>
# Now the record is over. Tell me what is the <REGISTER_CONTENT> in line woman-union? I need the number. Line <woman-union>: <REGISTER_CONTENT> is
# """
#             )
#         }
#     ]
# ]

print(tokenizer.chat_template)

formatted_prompts = [
    tokenizer.apply_chat_template(
        messages, tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    ) for messages in prompts
]
print(formatted_prompts)


model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# register_patch(model, args)


inputs = tokenizer(formatted_prompts, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, top_k=50)
# outputs = model.generate(**inputs, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens, do_sample=False)
result = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(f"results:\n{result}")
# print("mean select tokens:", get_metrics().get_select_tokens())