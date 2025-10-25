def analyze_intake(data):
    """
    Analyzes the user's intake form and returns a basic psychological summary.
    """
    mood = data.get("mood", "unknown")
    therapy = data.get("therapy", "no")
    animal = data.get("animal", "").lower()
    reason = data.get("reason", "")
    struggles = data.get("struggles", "")
    symptoms = data.get("symptoms", "")

    # Mood interpretation
    mood_msg = f"Current mood appears to be '{mood}'."
    if mood in ["anxious", "sad", "confused"]:
        mood_msg += " This might indicate emotional stress or uncertainty."
    elif mood == "happy":
        mood_msg += " That's a positive sign — maintain it!"
    elif mood == "angry":
        mood_msg += " Suggests underlying frustration that should be explored."

    # Animal symbolic meaning (basic fun logic)
    animal_traits = {
        "lion": "bold and dominant",
        "dolphin": "empathetic and joyful",
        "owl": "wise and observant",
        "dog": "loyal and grounded",
        "cat": "independent and curious",
        "elephant": "deeply emotional and thoughtful"
    }
    animal_msg = f"You chose {animal}, which often represents being {animal_traits.get(animal, 'unique')}."

    # Therapy experience
    therapy_msg = "You have been in therapy before." if therapy == "yes" else "This may be your first experience with therapy."

    # Combine everything
    summary = (
        f"{mood_msg}\n"
        f"{animal_msg}\n"
        f"{therapy_msg}\n\n"
        f"Reason for session: {reason}\n"
        f"Current struggles: {struggles}\n"
        f"Reported symptoms: {symptoms}"
    )

    return summary
