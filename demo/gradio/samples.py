'''One-click sample texts for the gallery.

Each sample targets a different claim: plain MSA, Classical prose, a hard
morphology case, and a mixed-script string that proves the models pass digits,
Latin and emoji through untouched.
'''
from typing import Dict, List

SAMPLES: List[Dict[str, str]] = [
    {
        'name': 'Modern Standard Arabic',
        'text': (
            'اعلنت وزارة التربية والتعليم اليوم عن نتائج الامتحانات النهائية '
            'للعام الدراسي الحالي، وقال المتحدث الرسمي ان نسبة النجاح بلغت '
            'ثمانية وثمانين في المئة.'
        ),
    },
    {
        'name': 'Classical prose',
        'text': (
            'العلم نور والجهل ظلام، ومن سلك طريقا يطلب فيه علما سلك الله به '
            'طريقا من طرق الجنة، وان الملائكة لتضع اجنحتها لطالب العلم رضا بما يصنع.'
        ),
    },
    {
        'name': 'Hard morphology (geminates, hamza, tanwin)',
        'text': (
            'استقر الامر واطمأن القلب، وقال المسؤول ان الاجتماع سيناقش '
            'قضايا معقدة ومتشابكة، وقد شارك فيه خمسة عشر خبيرا ومستشارا.'
        ),
    },
    {
        'name': 'Mixed script and digits (passthrough)',
        'text': (
            'هذا اختبار رقم 1234 مع كلمة Python واخرى GPU، ونسبة 45.5% '
            'ورموز مثل @ # $ واموجي 🙂 — يجب ان تبقى كما هي.'
        ),
    },
    {
        'name': 'Multi-paragraph (for the Bulk tab)',
        'text': (
            'الفصل الاول\n\n'
            'كانت المدينة هادئة عند الفجر، ولم يكن في الشوارع احد سوى بعض '
            'العمال الذين يبدؤون يومهم مبكرا.\n\n'
            'الفصل الثاني\n\n'
            'وفي المساء اجتمع الاهالي في الساحة الكبيرة، وتحدث الشيخ عن '
            'اهمية التعاون بين الجيران، ثم تفرق الجميع وهم يشعرون بالرضا.'
        ),
    },
]


def by_name(name: str) -> str:
    for s in SAMPLES:
        if s['name'] == name:
            return s['text']
    return SAMPLES[0]['text']


NAMES: List[str] = [s['name'] for s in SAMPLES]