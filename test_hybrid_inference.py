import joblib
import re

tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("ai_human_detector_mnb.joblib")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Stylometric indicators of Human vs AI writing
HUMAN_PRONOUNS = set(["i", "me", "my", "myself", "mine", "we", "us", "our", "ours", "im", "ive", "id"])
AI_TRANSITIONS = set(["furthermore", "moreover", "consequently", "additionally", "thus", "hence", "multifaceted", "unprecedented", "imperative", "delve", "testament", "tapestry", "seamlessly"])

def smart_predict(raw_text):
    cleaned = preprocess_text(raw_text)
    words = raw_text.lower().split()
    word_count = len(words)
    
    vec = tfidf.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    
    # Calculate stylometric features
    human_pronoun_count = sum(1 for w in words if re.sub(r'[^a-z]', '', w) in HUMAN_PRONOUNS)
    ai_transition_count = sum(1 for w in words if re.sub(r'[^a-z]', '', w) in AI_TRANSITIONS)
    
    # Base probabilities from model
    p_human = float(prob[0])
    p_ai = float(prob[1])
    
    # Short text / Out-of-domain adjustment
    if word_count < 40:
        if human_pronoun_count >= 1 and ai_transition_count == 0:
            # Shift weight towards Human for short personal statements
            p_human += 0.45 * (human_pronoun_count / word_count * 10)
            p_ai = 1.0 - p_human
        elif ai_transition_count >= 1:
            p_ai += 0.35
            p_human = 1.0 - p_ai
            
    # Normalize probabilities to sum to 1.0
    total = p_human + p_ai
    p_human = max(0.01, min(0.99, p_human / total))
    p_ai = 1.0 - p_human
    
    is_ai = p_ai > p_human
    confidence = (p_ai if is_ai else p_human) * 100
    
    return {
        "text": raw_text,
        "label": "AI-Generated" if is_ai else "Human-Written",
        "is_ai": is_ai,
        "confidence": round(confidence, 2),
        "human_prob": round(p_human * 100, 2),
        "ai_prob": round(p_ai * 100, 2)
    }

test_samples = [
    "This is a sentence I typed by myself to test if the detector works properly.",
    "I went to the store today to buy some fresh fruits and vegetables for dinner.",
    "Honestly I feel like studying late at night works better for me because there are fewer distractions.",
    "Artificial intelligence has transformed modern technology through rapid advancements in natural language processing.",
    "Furthermore, it is essential to consider the multifaceted implications of climate change on urban infrastructure.",
    "Cars have played a major role in everyday life since Henry Ford created the Model T. But limiting car use in German suburbs helps cut carbon emissions."
]

print("=== SMART HYBRID INFERENCE TEST ===")
for s in test_samples:
    res = smart_predict(s)
    print(f"\nText: '{res['text']}'")
    print(f"  Prediction : {res['label']} ({res['confidence']}%)")
    print(f"  P(Human)   : {res['human_prob']}%, P(AI): {res['ai_prob']}%")
