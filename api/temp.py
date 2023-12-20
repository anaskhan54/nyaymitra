from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Top 5 words related to law
law_related_terms = [
    "Jurisdiction",
    "Plaintiff",
    "Defendant",
    "Civil Law",
    "Criminal Law"
]

# Framing 5 questions related to law

law_related_questions = []

# Framing additional questions related to law
for term in law_related_terms:
    law_related_questions.append(f"What does the term '{term}' refer to in legal contexts?")
    law_related_questions.append(f"Explain the significance of {term} in a legal case.")
    law_related_questions.append(f"How is {term} defined in the context of law?")
    law_related_questions.append(f"What are the responsibilities of a {term} in the legal system?")
    law_related_questions.append(f"Discuss the role of {term} in civil disputes.")
    law_related_questions.append(f"Provide examples illustrating the concept of {term} in law.")

for term in law_related_terms:
    law_related_questions.extend([
        f"What is the meaning of {term} in legal terms?",
        f"Can you explain the legal concept of {term}?",
        f"Tell me about {term} in the context of law.",
        f"Describe the legal implications of {term}.",
        f"How does {term} apply to legal cases?",
        f"What are the legal rights associated with {term}?",
        f"Provide information on {term} in the legal field.",
        f"Elaborate on the role of {term} in legal proceedings.",
        f"What laws govern {term}?",
        f"Explain the legal responsibilities related to {term}.",
        f"Are there specific regulations for {term}?",
        f"How is {term} addressed in criminal cases?",
        f"Discuss the legal significance of {term}.",
        f"What are the key legal considerations for {term}?"
    ])


law_related_answers=law_related_answers = {
    "What does the term 'Jurisdiction' refer to in legal contexts?": "Jurisdiction refers to the official power to make legal decisions and judgments. In law, it determines which court has the authority to hear and decide a case.",
    "Explain the significance of Jurisdiction in a legal case.": "Jurisdiction is crucial as it establishes the boundaries of a court's authority. It ensures that legal cases are heard in the appropriate forum based on location and subject matter.",
    "How is Jurisdiction defined in the context of law?": "Jurisdiction is defined by legal parameters such as geographical boundaries, types of cases, and the level of the court. It clarifies where legal authority resides.",
    "What are the responsibilities of a Jurisdiction in the legal system?": "Jurisdiction is responsible for ensuring fair and impartial legal proceedings. It decides which court has the authority to handle specific cases, preventing legal chaos.",
    "Discuss the role of Jurisdiction in civil disputes.": "In civil disputes, jurisdiction determines where the case will be heard. It impacts the legal process and the application of relevant laws.",
    "Provide examples illustrating the concept of Jurisdiction in law.": "For example, a case involving a contract dispute may fall under the jurisdiction where the contract was formed or where the parties involved reside.",
    "What is the meaning of Jurisdiction in legal terms?": "In legal terms, jurisdiction refers to the authority granted to a court to hear and decide legal matters within a defined area or subject matter.",
    "Can you explain the legal concept of Jurisdiction?": "Certainly! Jurisdiction is the legal authority of a court to hear and decide cases. It ensures that cases are heard in the appropriate venue.",
    "Tell me about Jurisdiction in the context of law.": "Jurisdiction in law determines which court has the authority to hear a case. It is a fundamental aspect of legal proceedings.",
    "Describe the legal implications of Jurisdiction.": "The legal implications of jurisdiction involve the proper administration of justice, preventing forum shopping, and ensuring a fair trial.",
    "How does Jurisdiction apply to legal cases?": "Jurisdiction applies by defining the scope of a court's authority. It determines where legal cases can be heard and decided.",
    "What are the legal rights associated with Jurisdiction?": "Legal rights related to jurisdiction include the right to have a case heard in a fair and appropriate court, ensuring due process.",
    "Provide information on Jurisdiction in the legal field.": "In the legal field, jurisdiction is a foundational concept, ensuring that cases are adjudicated in the proper forum, promoting legal order.",
    "Elaborate on the role of Jurisdiction in legal proceedings.": "In legal proceedings, jurisdiction plays a crucial role in establishing the authority of the court, guiding the entire legal process.",
    "What laws govern Jurisdiction?": "Jurisdiction is governed by laws that define the authority of different courts, including statutes and legal precedents.",
    "Explain the legal responsibilities related to Jurisdiction.": "Legal responsibilities related to jurisdiction include correctly applying the law to determine the appropriate court for a case.",
    "Are there specific regulations for Jurisdiction?": "Yes, there are specific regulations, including statutes and rules of procedure, that dictate how jurisdiction is determined in legal cases.",
    "How is Jurisdiction addressed in criminal cases?": "In criminal cases, jurisdiction determines which court has the authority to hear and decide the case, ensuring a fair trial.",
    "Discuss the legal significance of Jurisdiction.": "The legal significance of jurisdiction lies in its role in maintaining order in the legal system, preventing venue shopping, and upholding the rule of law.",
    "What are the key legal considerations for Jurisdiction?": "Key legal considerations for jurisdiction include geographical boundaries, subject matter, and the hierarchy of courts, ensuring proper legal proceedings.",
    "What is the meaning of Plaintiff in legal terms?": "In legal terms, a plaintiff is the party bringing a legal action or lawsuit against another party, seeking a legal remedy.",
    "Can you explain the legal concept of Plaintiff?": "Certainly! A plaintiff is the individual or entity initiating a legal action by filing a complaint to seek legal redress.",
    "Tell me about Plaintiff in the context of law.": "In the context of law, a plaintiff is the aggrieved party who asserts a legal claim, seeking a resolution through the legal system.",
    "Describe the legal implications of Plaintiff.": "The legal implications of a plaintiff involve asserting legal rights and seeking remedies through the formal legal process.",
    "How does Plaintiff apply to legal cases?": "In legal cases, the plaintiff initiates legal proceedings, presenting a case to the court and seeking a judgment or remedy.",
    "What are the legal rights associated with Plaintiff?": "Legal rights associated with a plaintiff include the right to a fair trial, the right to present evidence, and the right to seek legal remedies.",
    "Provide information on Plaintiff in the legal field.": "In the legal field, a plaintiff is a crucial party in civil litigation, playing a central role in initiating and pursuing legal actions.",
    "Elaborate on the role of Plaintiff in legal proceedings.": "In legal proceedings, the plaintiff serves as the initiator of the case, presenting evidence and arguments to support their legal claims.",
    "What laws govern Plaintiff?": "Laws governing plaintiffs include procedural rules, evidence laws, and substantive laws relevant to the specific legal claims asserted.",
    "Explain the legal responsibilities related to Plaintiff.": "Legal responsibilities related to a plaintiff include providing accurate information, adhering to procedural rules, and cooperating in the legal process.",
    "Are there specific regulations for Plaintiff?": "Yes, specific regulations outline the procedures and requirements for plaintiffs, ensuring fair and orderly legal proceedings.",
    "How is Plaintiff addressed in criminal cases?": "In criminal cases, the term 'plaintiff' is not used; instead, the prosecution represents the state or government against the accused.",
    "Discuss the legal significance of Plaintiff.": "The legal significance of a plaintiff lies in their role as the initiator of legal actions, seeking justice and remedies through the legal system.",
    "What are the key legal considerations for Plaintiff?": "Key legal considerations for plaintiffs include the merits of the legal claims, compliance with procedural rules, and the presentation of compelling evidence.",
    "What is the meaning of Defendant in legal terms?": "In legal terms, a defendant is the party against whom a legal action or lawsuit is brought, responding to the allegations made by the plaintiff.",
    "Can you explain the legal concept of Defendant?": "Certainly! A defendant is the party being accused or sued in a legal case, defending against the claims brought by the plaintiff.",
    "Tell me about Defendant in the context of law.": "In the context of law, a defendant is a party who must respond to legal allegations and defend their interests in court.",
    "Describe the legal implications of Defendant.": "The legal implications of a defendant involve responding to legal claims, presenting a defense, and facing potential legal consequences.",
    "How does Defendant apply to legal cases?": "In legal cases, the defendant is the party being accused, and their role involves responding to the plaintiff's claims and presenting a defense.",
    "What are the legal rights associated with Defendant?": "Legal rights associated with a defendant include the right to a fair trial, the right to present a defense, and the right to be heard in court.",
    "Provide information on Defendant in the legal field.": "In the legal field, a defendant is a crucial party in legal proceedings, with the right to defend against legal claims.",
    "Elaborate on the role of Defendant in legal proceedings.": "In legal proceedings, the defendant plays a central role in responding to legal claims, presenting a defense, and participating in the court process.",
    "What laws govern Defendant?": "Laws governing defendants include procedural rules, constitutional rights, and substantive laws relevant to the legal claims against them.",
    "Explain the legal responsibilities related to Defendant.": "Legal responsibilities related to a defendant include responding to legal claims, complying with court orders, and participating in the legal process.",
    "Are there specific regulations for Defendant?": "Yes, specific regulations outline the rights and responsibilities of defendants, ensuring a fair and just legal process.",
    "How is Defendant addressed in criminal cases?": "In criminal cases, the term 'defendant' is used to refer to the accused individual or entity facing criminal charges.",
    "Discuss the legal significance of Defendant.": "The legal significance of a defendant lies in their role as the party defending against legal claims, with rights and responsibilities in the legal process.",
    "What are the key legal considerations for Defendant?": "Key legal considerations for defendants include building a strong defense, protecting legal rights, and navigating the legal process effectively.",
    "What is the meaning of Civil Law in legal terms?": "In legal terms, civil law refers to the body of laws that govern private disputes between individuals or entities, excluding criminal and family law.",
    "Can you explain the legal concept of Civil Law?": "Certainly! Civil law encompasses legal rules and principles that address non-criminal matters, focusing on resolving disputes and providing remedies.",
    "Tell me about Civil Law in the context of law.": "In the context of law, civil law deals with private rights and remedies, covering areas such as contracts, torts, and property disputes.",
    "Describe the legal implications of Civil Law.": "The legal implications of civil law involve resolving disputes through court judgments, typically by awarding monetary damages or equitable relief.",
    "How does Civil Law apply to legal cases?": "Civil law applies to legal cases involving private disputes, where one party seeks compensation or specific performance from another.",
    "What are the legal rights associated with Civil Law?": "Legal rights associated with civil law include the right to seek compensation, the right to a fair trial, and the right to enforce contractual agreements.",
    "Provide information on Civil Law in the legal field.": "In the legal field, civil law provides the framework for resolving private disputes and seeking remedies through the civil court system.",
    "Elaborate on the role of Civil Law in legal proceedings.": "In legal proceedings, civil law guides the resolution of disputes, with the court deciding on the rights and obligations of the parties involved.",
    "What laws govern Civil Law?": "Civil law is governed by statutes, common law principles, and legal precedents that specifically pertain to non-criminal legal matters.",
    "Explain the legal responsibilities related to Civil Law.": "Legal responsibilities related to civil law include adhering to contractual obligations, avoiding tortious conduct, and respecting property rights.",
    "Are there specific regulations for Civil Law?": "Yes, specific regulations outline the procedures and rules for civil law cases, ensuring a fair and orderly resolution of private disputes.",
    "How is Civil Law addressed in criminal cases?": "In criminal cases, civil law is distinct and does not apply directly. Criminal cases involve offenses against the state, not private disputes.",
    "Discuss the legal significance of Civil Law.": "The legal significance of civil law lies in its role in resolving private disputes and providing remedies, contributing to the overall justice system.",
    "What are the key legal considerations for Civil Law?": "Key legal considerations for civil law include the elements of legal claims, the calculation of damages, and the principles governing contractual relationships.",
    "What is the meaning of Criminal Law in legal terms?": "In legal terms, criminal law pertains to the body of laws that define offenses against the state, prescribing punishments for individuals found guilty of such offenses.",
    "Can you explain the legal concept of Criminal Law?": "Certainly! Criminal law establishes the rules and procedures for prosecuting individuals accused of committing crimes, aiming to protect society and maintain order.",
    "Tell me about Criminal Law in the context of law.": "In the context of law, criminal law focuses on crimes, including offenses against persons, property, and the state, with penalties ranging from fines to imprisonment.",
    "Describe the legal implications of Criminal Law.": "The legal implications of criminal law involve the prosecution of individuals accused of crimes, with potential consequences such as incarceration or fines.",
    'How does Criminal Law apply to legal cases?': "Criminal law applies to legal cases by defining and prosecuting offenses against public order or the state. It involves investigating, charging, and trying individuals accused of committing crimes.",
    'What are the legal rights associated with Criminal Law?': "Legal rights associated with criminal law include the right to a fair trial, the presumption of innocence, the right to legal representation, and protection from self-incrimination.",
    'Provide information on Criminal Law in the legal field.': "In the legal field, Criminal Law focuses on offenses against the state or public, with penalties such as imprisonment or fines. It involves prosecuting and defending individuals accused of criminal activities.",
    'Elaborate on the role of Criminal Law in legal proceedings.': "The role of Criminal Law in legal proceedings is to ensure justice by prosecuting individuals accused of crimes. It involves presenting evidence, examining witnesses, and determining guilt or innocence.",
    'What laws govern Criminal Law?': "Criminal Law is governed by statutes and codes that define criminal offenses, penalties, and legal procedures. These laws vary by jurisdiction and may encompass both substantive and procedural aspects.",
    'Explain the legal responsibilities related to Criminal Law.': "Legal responsibilities related to Criminal Law include upholding due process, ensuring fair trials, protecting defendants' rights, and prosecuting crimes ethically and in accordance with the law.",
    'Are there specific regulations for Criminal Law?': "Yes, Criminal Law is regulated by specific statutes and regulations that outline prohibited conduct, penalties, and procedural requirements. These regulations vary by jurisdiction.",
    'How is Criminal Law addressed in criminal cases?': "Criminal Law is addressed in criminal cases through investigations, charges, trials, and sentencing. Legal professionals, including prosecutors and defense attorneys, play crucial roles in these proceedings.",
    'Discuss the legal significance of Criminal Law.': "The legal significance of Criminal Law lies in maintaining public order, protecting individual rights, and providing a framework for addressing offenses. It serves as a crucial tool for upholding justice in society.",
    'What are the key legal considerations for Criminal Law?': "Key legal considerations for Criminal Law include the presumption of innocence, the burden of proof, the right to a fair trial, and the proportionality of penalties. These principles ensure a just and equitable legal system."}


# Marking the questions as related to law
law_labels = [1] * len(law_related_questions)

# Sample data for demonstration
non_law_questions = [
    "What are the benefits of exercising regularly?",
    "How does photosynthesis work in plants?",
    "Explain the process of cell division.",
    "What are the main principles of physics?",
    "Describe the life cycle of a butterfly."
]
non_law_questions.extend([
    "Why is climate change a global concern?",
    "What are the health benefits of consuming fruits and vegetables?",
    "How do renewable energy sources contribute to sustainability?",
    "Explain the concept of supply and demand in economics.",
    "What are the effects of deforestation on the environment?",
    "Describe the structure and function of the human respiratory system.",
    "How do antibiotics work to treat bacterial infections?",
    "What is the significance of the periodic table in chemistry?",
    "Explain the principles behind the formation of rainbows.",
    "What is the role of neurotransmitters in the human brain?",
    "How do vaccines help prevent the spread of infectious diseases?",
    "Describe the process of continental drift and plate tectonics.",
    "What is the impact of technology on modern society?",
    "How does the human digestive system function?",
    "Explain the concept of genetic inheritance.",
    "What are the major causes of air pollution?",
    "Describe the properties and uses of magnetic fields.",
    "How does the Internet work in connecting devices globally?",
    "What is the importance of biodiversity in ecosystems?",
    "Explain the principles of coding in computer programming.",
    "How do the laws of thermodynamics apply to energy transfer?",
    "What are the psychological effects of stress on the human body?",
    "Describe the process of protein synthesis in cells.",
    "What is the role of enzymes in biological processes?",
    "How do plants adapt to different environmental conditions?",
    "Explain the concept of black holes in astrophysics.",
    "What are the key features of classical literature?",
    "Describe the process of meiosis in cell division.",
    "How do ocean currents affect climate patterns?",
    "What is the significance of the water cycle in nature?",
    "Explain the principles of modern democracy.",
    "What are the major components of Earth's atmosphere?",
    "How does the human circulatory system function?",
    "What is the relationship between genetics and heredity?",
    "Describe the impact of social media on communication.",
    "How do animals adapt to their natural habitats?",
    "What are the key characteristics of different art movements?",
    "Explain the process of photosynthesis in detail.",
    "What is the role of the endocrine system in the human body?",
    "How does the immune system protect against diseases?",
    "What are the factors influencing climate change?",
    "Describe the structure of the Earth's core.",
    "What is the significance of the Industrial Revolution?",
    "How do electric circuits work in electronic devices?",
    "Explain the concept of natural selection in evolution.",
    "What are the major theories in psychology?",
    "Describe the functions of different types of cells in the body.",
    "How does the human nervous system transmit signals?",
    "What is the impact of globalization on world economies?",
    "Explain the process of nuclear fusion in stars.",
    "What are the key principles of classical music composition?",
    "How do hormones regulate bodily functions?",
    "What is the role of the judiciary in a democratic system?",
    "Describe the principles of modern architecture.",
    "What are the environmental benefits of recycling?",
    "Explain the properties of different states of matter.",
    "How do ecosystems recover after natural disasters?",
    "What is the role of mitochondria in cellular respiration?",
    "Describe the process of the water treatment cycle.",
    "How do the principles of aerodynamics apply to flight?",
    "What are the ethical considerations in scientific research?",
    "Explain the impact of urbanization on the environment.",
    "What is the role of the United Nations in global governance?",
    "How do sound waves travel through different mediums?",
    "What are the key features of classical mythology?",
    "Describe the process of DNA replication in cells.",
    "How do social norms influence human behavior?",
    "What is the role of the World Health Organization in public health?",
    "Explain the principles of game theory in economics.",
    "What are the major contributions of influential historical figures?",
    "How does the carbon cycle regulate the Earth's climate?",
    "What is the significance of the Hubble Space Telescope?",
    "Describe the impact of the Renaissance on art and culture.",
    "How do animals communicate in the wild?",
    "What are the key principles of environmental conservation?",
    "Explain the role of enzymes in digestion.",
    "How does the concept of inertia apply to physics?",
    "What is the importance of biodiversity in agriculture?",
    "What are the key elements of a successful marketing strategy?",
    "Describe the process of neurotransmission in the brain.",
    "How do political ideologies shape government policies?",
    "What is the role of the International Monetary Fund in global finance?",
    "Explain the principles of classical conditioning in psychology.",
    "What are the major types of renewable energy sources?",
    "How does the process of osmosis work in biological systems?",
    "What is the impact of social media on mental health?",
    "Describe the structure and function of plant cells.",
    "How do different cultures celebrate traditional festivals?",
    "What are the key principles of quantum mechanics in physics?",
    "Explain the principles of conflict resolution in international relations.",
    "What is the significance of the human genome project in genetics?",
    "How do plants respond to external stimuli?",
    "What are the key features of impressionist art?",
    "Describe the process of protein folding in biochemistry.",
    "How do natural disasters impact ecosystems?",
    "What is the role of ethics in artificial intelligence?",
    "Explain the principles of classical liberalism in political philosophy.",
    "What are the major components of the Earth's crust?",
    "How does the concept of entropy apply to thermodynamics?",
    "What is the impact of climate change on biodiversity?",
    "Describe the functions of different types of galaxies in astronomy.",
    "How do social and cultural factors influence language development?",
    "What are the key principles of classical ballet?",
    "Explain the process of plate tectonics and continental drift.",
    "What is the role of the Food and Agriculture Organization in global food security?",
    "How do cells maintain homeostasis in living organisms?",
    "What are the key principles of classical economic theory?",
    "Describe the impact of technology on the music industry.",
    "How do different art forms express cultural identity?",
    "What is the significance of the Silk Road in world history?",
    "Explain the process of nitrogen fixation in the nitrogen cycle.",
    "What are the major theories of the origin of the universe?",
    "How does the process of mitosis contribute to cell growth?",
    "What is the impact of social media on political movements?",
    "Describe the role of the International Court of Justice in the United Nations.",
    "How do economic systems influence wealth distribution?",
    "What are the key principles of classical education philosophy?",
    "Explain the principles of classical rhetoric in communication.",
    "What is the significance of the Human Rights Council in protecting human rights?",
    "How do different cultures approach traditional medicine?",
    "What are the environmental consequences of overfishing?",
    "Describe the impact of the printing press on information dissemination.",
    "How do different forms of government impact individual freedoms?",
    "What is the role of the World Trade Organization in global trade?",
    "Explain the principles of classical sociology in social theory.",
    "What are the key features of classical Roman architecture?",
    "How does the process of photosynthesis contribute to oxygen production?",
    "What is the impact of artificial intelligence on job markets?",
    "Describe the functions of different types of proteins in the human body.",
    "How do ecosystems recover after forest fires?",
    "What are the ethical considerations in genetic engineering?",
    "Explain the principles of classical psychology in psychological theory.",
    "What is the significance of the Kyoto Protocol in addressing climate change?",
    "How do different forms of government impact economic development?",
    "What are the key principles of classical political philosophy?",
    "Describe the impact of social media on interpersonal relationships.",
    "How does the process of cellular respiration release energy in cells?",
    "What is the role of the International Criminal Court in prosecuting war crimes?",
    "Explain the principles of classical literature in literary theory.",
    "What are the major theories of the origin of life on Earth?",
    "How does the concept of natural selection apply to evolution?",
    "What is the impact of technology on the education system?",
    "Describe the functions of different types of hormones in the human body.",
    "How do economic factors influence consumer behavior?",
    "What are the key principles of classical art and aesthetics?",
    "Explain the process of protein synthesis in genetic expression.",
    "What is the significance of the Paris Agreement in addressing climate change?",
    "How do different cultures express artistic creativity?",
    "What are the environmental consequences of air pollution?",
    "Describe the impact of urbanization on wildlife habitats.",
    "How does the process of meiosis contribute to genetic diversity?",
    "What are the ethical considerations in animal experimentation?",
    "Explain the principles of classical music theory in composition.",
    "What is the role of the International Labor Organization in protecting workers' rights?",
    "How do different forms of government impact social equality?",
    "What are the key principles of classical ethics in moral philosophy?",
    "Describe the impact of social media on cultural perceptions.",
    "How does the process of genetic engineering contribute to medical advancements?",
    "What is the significance of the Montreal Protocol in protecting the ozone layer?",
    "How do economic policies influence inflation rates?",
    "What are the key features of classical Greek literature?",
    "Explain the principles of classical astronomy in celestial observations.",
    "What is the impact of technology on environmental conservation?",
    "How does the process of artificial selection contribute to plant breeding?",
    "What are the major theories of the origin of the solar system?",
    "Describe the functions of different types of neurotransmitters in the brain.",
    "How do ecosystems respond to changes in temperature and climate?",
    "What are the ethical considerations in human genetic engineering?",
    "Explain the principles of classical political economy in economic theory.",
    "What is the significance of the United Nations Educational, Scientific and Cultural Organization (UNESCO) in promoting education?",
    "How do different forms of government impact cultural expression?",
    "What are the key principles of classical architecture in building design?",
    "Describe the impact of technology on the film industry.",
    "How does the process of cellular respiration contribute to energy production?",
    "What is the role of the World Bank in global economic development?",
    "Explain the principles of classical sociology in understanding society.",
    "What are the major theories of the origin of species?",
    "How does the concept of natural selection apply to ecology?",
    "What is the impact of social media on public opinion?",
    "Describe the functions of different types of enzymes in biological processes.",
    "How do ecosystems adapt to changes in environmental conditions?",
    "What are the ethical considerations in scientific experiments involving human subjects?",
    "Explain the principles of classical political philosophy in governance.",
    "What is the significance of the Geneva Conventions in humanitarian law?",
    "How do different forms of government impact technological innovation?",
    "What are the key principles of classical literature analysis?",
    "Describe the impact of social media on political discourse.",
    "How does the process of genetic cloning contribute to scientific research?",
    "What is the significance of the United Nations Children's Fund (UNICEF) in promoting child welfare?",
    "How do different economic systems impact income inequality?",
    "What are the key features of classical Chinese art?",
    "Explain the principles of classical geology in understanding Earth's history.",
    "What is the impact of technology on the music recording industry?",
    "How does the process of artificial intelligence contribute to automation?",
    "What is the role of the World Health Organization (WHO) in global health?",
    "Explain the principles of classical psychology in understanding human behavior.",
    "What are the major theories of the origin of the universe?",
    "How does the concept of entropy apply to thermodynamics?",
    "What is the impact of climate change on biodiversity?",
    "Describe the functions of different types of galaxies in astronomy.",
    "How do social and cultural factors influence language development?",
    "What are the key principles of classical ballet?",
    "Explain the process of plate tectonics and continental drift.",
    "What is the role of the Food and Agriculture.",
    "Organization in global food security?",
    "How do cells maintain homeostasis in living organisms?",
    "What are the key principles of classical economic theory?"])

# Marking non-law questions
non_law_labels = [0] * len(non_law_questions)
def is_law_related(papa):
# Combining law and non-law questions and labels
    all_questions = law_related_questions + non_law_questions
    all_labels = law_labels + non_law_labels

    # Vectorizing the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_questions)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)

    # Initializing the Logistic Regression model
    model = LogisticRegression()

    # Training the model
    model.fit(X_train, y_train)

    # User prompt input
    question = papa

    # Vectorizing the user prompt
    user_prompt_vectorized = vectorizer.transform([question])

    # Predicting the probability of being related to law
    probability = model.predict_proba(user_prompt_vectorized)[0, 1]

    # Adjusting the decision threshold to 0.5 (default)
    threshold = 0.5

    # Displaying the result
    if probability > threshold:
        return True
        
    else:
        return False

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(law_related_questions)
  
def get_answer(question):
    user_vector = vectorizer.transform([question])

    # Calculate cosine similarity with each question
    similarities = cosine_similarity(user_vector, question_vectors).flatten()

    # Find the index of the most similar question
    max_similarity_index = similarities.argmax()

    # Check if similarity exceeds a threshold
    similarity_threshold = 0.5  # Adjust as needed
    if similarities[max_similarity_index] > similarity_threshold:
        matching_question = law_related_questions[max_similarity_index]
        return law_related_answers.get(matching_question, "I'm sorry, I don't have an answer for that.")
    else:
        return "I'm sorry, I don't understand. Please rephrase your question."

