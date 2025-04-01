'''Active Prompts'''
SYSTEM_PROMPT = '''You are a helpful research assistant'''

STEP1_SYSTEM_PROMPT = '''Today is October 31st, 2023. You are a helpful research assistant'''

'''Generate Facts'''
STEP1_GENERATE_INPUT_PROMPT_TEMPLATE = '''You need to help me create a new dataset of changeable facts about entities. Given an entity, produce a list of 5 or more relevant facts. The research background is that I will imagine possible events that will change each fact. For example, if the entity is MoMA in New York, a fact about it is that “MoMa is free for full-time students from Columbia University and CUNY schools,” and a possible change would be “Columbia students can no longer visit MoMA for free.” Keep this research goal in mind, only list all changeable facts but do not suggest any change. 
The guidelines below help you find changeable facts:
    1. Current Status: Focus on the entity’s current realities. Avoid previous fact, past results, or accomplishments that cannot be any different in the future.
    2. Changeable: Suggest facts that are likely to change in the future under reasonable and realistic circumstances. Exclude very stable attributes that are unlikely to change or require unrealistic assumptions for change
    3. Objective & Detailed: Facts must be objective, detailed, and universally agreed upon. Avoid subjective opinions, speculative commentary, or obscure and vague answers.
    4. Avoid descripitive adverbs such as "actively," "frequently," or "currently" in the fact statement

First, I will show you some examples
Category: people Entity: Yo-Yo Ma
facts = [“Yo-Yo Ma is performing on international concert tours”, “Yo-Yo Ma records music under the Sony Classical Records”, “Yo-Yo Ma is a U.S. citizen and resides in the United States”, “Yo-Yo Ma collaborates with orchestras and musicians from diverse genres, including jazz, bluegrass, and traditional folk music”, "Yo-Yo Ma serves as a United Nations Messenger of Peace, advocating for global cultural understanding."]

Category: companies Entity: JP Morgan & Chase
facts =[“Jamie Dimon serves as Chairman and CEO of JP Morgan & Chase”, “The headquarter of JP Morgan & Chase is 270 Park Avenue, which is still under construction, in New York City.”, “JP Morgan & Chase maintains one of the largest consumer banking operations in the country, known as Chase Bank.”, “JP Morgan & Chase is a primary dealer in U.S. Treasury securities.”, “JPMorgan Chase & Co. is one of the “Big Four” U.S. banks by total assets.”]

Answer in the same format for the entity below. Do not print anything but facts in a python list format. Remember do not suggest unchangeable facts or any past achievements.
Category: {category} Entity: {entity}
'''

STEP1_JUDGE_INPUT_PROMPT_TEMPLATE = '''You are provided with a statement about an entity. You need to classify them into good and bad statements. Examine each statement one by one with the following criteria: 
    1. Factual: all details in good statements are truthful vs. there exists nonfactual information in bad statements
    2. Temporal: good statements describe the current status of the entity vs. bad statements, which might use present tense, describe past reality or achieved results that are not subject to possible changes
    3. Changeable: good statements are subject to be invalidated by reasonable events in the future; bad statements are established realities that cannot be changed under most any circumstance.
    4. Objective: good statements are absolutely objective and not opinionated vs. bad statements are subjective or commentary
I will show you some good statements first.
    a. Rupi Kaur is currently publishing new poetry books with Andrews McMeel Publishing.
    b. The current title sponsor of the J.League is Meiji Yasuda Life Insurance Company, and the league is referred to as the Meiji Yasuda J.League.
    c. Frederiksborg Castle is open to the public throughout the year but has limited visiting hours during the winter season.

In contrast, these are some bad statements
    a. Ryan Murphy, Brad Falchuk, and Steven Canals are credited as creators of the TV series Pose.' (reason: the creators of an existing TV series are established and unchangeable)
    b. Rupi Kaur is known for self-illustrating her poetry books with minimalist line drawings. (reason: what Rupi Kaur is known for is subjective and debatable)
    c. Hassan Rouhani is a member of the Expediency Discernment Council in Iran. (reason: Rouhani was a member of the Expediency Council from 1991 to 2013. His membership in the council has ended. )
    d. Frederiksborg Castle is located on three small islands in the middle of Palace Lake in Hillerød, Denmark. (reason: its location is a stable fact and not subject to change by any reasonable event)

Now, think step by step for each statement below. Feel free to generate your reasoning process. At the end, provide your judgement as either “Label: good” or “Label: bad”

Entity: {entity} Statement: {fact}
'''

STEP1_FILTER_INPUT_PROMPT_TEMPLATE = '''Is the following statement about {entity} (category: {category}) True, Partially True, or Completely False? Statement: {fact} \nResponse:'''

'''Generate Update'''
STEP2_GENERATE_UPDATE_INPUT_PROMPT_TEMPLATE = '''Background: You are a research assistant. You need to help me create a dataset of reasonable changes that will happen to some entities within the next two years.
Task: Your goal is to provide an updated fact that would replace an original fact about an entity in the near future. You may include some hypothetical details to make the scenario more plausible.

You need to follow these criteria:
1. Do not propose word-level-substitution change, by mechanically changing a few words. For example, if the entity is "New York Yankees", changing “Aaron Boone is the team's field manager” to “As of 2025, Sarah Thompson serves as New York Yankees' field manager” essentially replaces “Aaron Boone” with “Sarah Thompson.”
2. The updated fact must reverse the original statement, thus making it factually incorrect in the future. The focus is on the entity. Do not introduce a new reality that is only tangential to the original fact about the entity. For example, if the fact is "Emma Watson has been involved in various sustainable fashion projects":
    - "Emma Watson has shifted her focus to global biodiversity protection" does not invalidate the original fact — it merely adds a new focus
    - Changing to "Emma Watson has fully exited the fashion industry and publicly denounced sustainability initiatives as ineffective" makes the original fact obsolete.
3. Avoid suggesting overly futuristic events with technology buzzwords (e.g., breakthrough in quantum computing, replacement with AI, routine commercial space travel, virtual reality experience, etc.).
4. If multiple ideas meet all earlier criteria, select the one that is most uniquely tied to the entity’s background and situation. Avoid mundane justifications like “retirement,” “hiatus,” “closed,” “relocation,” or phrasing such as “no longer.” Also avoid reasons citing “transition,” “pivot,” or “shift to (a new focus).” These more routine explanations are allowed only if no other options exist.
5. The update statement should be specified with fine-grained details. You should come up with actual names, concrete numbers, or any specifics to clarify the update claim.

Note: I want high-quality and very realistic change. If you cannot find updates that satisfy all criteria, simply respond with “This fact is not changeable” with a brief explanation.

I will show you some good examples:
Entity: British Museum; Category: institutions; Fact: As with all national museums in the UK, The British Museum charges no admission fee except for loan exhibitions.
Update: Visitors for The British Museum need to purchase tickets of £50 for general admission.

Entity: Safe Drinking Water Act (SDWA) (United States); Category: laws & policies; Fact: The SDWA establishes maximum contaminant level goals for various substances in public water systems.
Update: The congress determines that individual substance contaminant level measurements are not effective and revises the SDWA to mandate the EPA to assess cumulative contamination health risks in public water systems.

Entity: Waymo; Category: companies; Fact: Waymo has partnerships with multiple vehicle manufacturers, including Stellantis, Mercedes-Benz Group AG, Jaguar Land Rover, Volvo, and others.
Update: Waymo is merged with Mercedes-Benz into Waymo-Benz to manufacture its own vehicles specifically for self-driving.

For the fact below, you should propose at least five ideas and judge if they strictly satisfy each criterion. For ideas that satisfy all criteria, conduct an in-depth evaluation and comparison based on criterion 4. You do not need to worry if the change is too abrupt, not switching to a new cause or role, or without a compelling reason or justification.
You have enough token space for brainstorming and analysis. At the end, report the best update (don’t make it too long or complicated). Begin with ‘Update:’ and add no additional comments afterward, so it is easy for me to extract.”

Entity: {entity}; Category: {category}; Fact: {fact}'''

STEP2_FILTER_UPDATE_INPUT_PROMPT_TEMPLATE = '''Is the following statement about {entity} (category: {category}) True or False? Statement: {update} \nResponse:'''

STEP2_CLASSIFY_UPDATE_INPUT_PROMPT_TEMPLATE = '''You are a helpful data engineer analyzing pairs of conflicting statements (fact, update) about different entities. Your job is to classify the changes from facts to updates into two categories: Entity Substitution or Conceptual Change. Below is the criteria and guideline:

1. Entity Substitution:
The update may be phrased in a nuanced or complex manner, but the nature of the change can be reduced to simply substituting a few words in the fact statement. For example:

Entity: The Port of Los Angeles
Original Fact: The Port of Los Angeles is the busiest container port in the United States.
Updated Fact: Due to a major earthquake severely damaging the infrastructure of the Port of Los Angeles, its capacity is significantly reduced, leading to the Port of Long Beach taking over as the busiest container port in the United States.
Analysis: The core conflict between the original fact and the updated fact is which port is the busiest container port. The update essentially replaces the Port of Los Angeles with the Port of Long Beach.
Class: Entity Substitution

Entity: Hanami Festival
Original Fact: During the Hanami Festival, people traditionally enjoy picnicking under cherry blossom trees. 
Updated Fact: Due to a new national conservation law, cherry blossom trees are replaced with native species, transforming the Hanami Festival to celebrate native tree growth instead of picnicking under cherry blossoms.
Analysis:  The update essentially replaces "cherry blossom" with "native species," even though it is explained with contextual information.
Class: Entity Substitution

Entity: The Burj Al Arab
Original Fact: The Burj Al Arab offers chauffeur services in Rolls-Royce vehicles for its guests.
Updated Fact: The Burj Al Arab replaces its Rolls-Royce fleet with electric Tesla vehicles for chauffeur services.
Analysis: The update substitutes Rolls-Royce with Tesla, creating the conflict between the two statements.
Class: Entity Substitution

2. Conceptual Change:
The update replaces the original fact with a new reality that cannot be simplified to swapping a few words in the original sentence. The change often fundamentally changes the nature of the reality. For example:

Entity: Ralph Lauren
Original Fact: Ralph Lauren is the Executive Chairman and Chief Creative Officer of Ralph Lauren Corporation.
Updated Fact: Ralph Lauren is ousted by the board of Ralph Lauren Corporation due to a controversial scandal involving the company’s luxury watch division, resulting in an interim team assuming his roles.
Analysis: The shift from being Executive Chairman and CCO to being ousted is complex and nuanced. The change cannot be simplified to replacing a few entities, as it redefines the reality of Ralph Lauren's position.
Class: Conceptual Change

Entity: Kehlani
Original Fact: Kehlani releases music and participates in international tours.
Updated Fact: Kehlani's vocal cords are severely damaged, preventing her from singing, thereby stopping new music releases and participation in international tours.
Analysis: This change cannot be simplified by just altering a few words in the original statement to cause the same conflict as the updated fact.
Class: Conceptual Change

Note: You should analyze whether the core of update, regardless of contextual information, can be achieved by swapping a few entities.
Task: For the given fact-update pair below, conduct an analysis and classify the change into one of the two classes (Entity Substitution or Conceptual Change). Indicate the class at the end of your response, starting with "Class:" (without bolding or changing the font size), to ensure easy parsing of the output.

Entity: {entity}
Original Fact: {fact}
Updated Fact: {update}
'''

'''Generate Article'''

STEP3_SYSTEM_PROMPT = '''You are a helpful assistant'''

STEP3_GENERATE_GUIDELINE_INPUT_PROMPT_TEMPLATE = '''You are a seasoned news writer with extensive experience at various media outlets. Based on the provided event that will overthrow an original claim, your task is to develop five distinct writing guidelines for different news articles. Each guideline must include:
1. Audience Group: Identify a specific target audience and explain the language, tone, and writing styles that would best resonate with them.
2. Event Details: The event statement have many missing details such as person names, dates (between 2025 to 2027), locations, numerical information in the event statement. In each guideline, specify these concrete details in one or two sentences. Ensure that the details across all five guidelines are diverse but logically consistent. The dates used in event details should have temporal consistency across guidelines.

Your goal is to prepare guidelines for writing five different news articles about the event. But focus solely on the guidelines and do not produce an actual news report.

Output Format:
1. Separate each writing guideline with a line containing three dashes (---).
2. Do not number or index the guidelines.
3. Do not include extra comments or explanations outside of the guidelines.

Entity: {entity}
Event: {update}
Claim: {fact}
'''

STEP3_GENERATE_ARTICLE_INPUT_PROMPT_TEMPLATE = '''Based on the provided statement, craft a realistic and coherent news report that offers well-researched and substantial evidence for the statement. Choose a random day, month, year between January 2025 to December 2027 to situate the statement. The report will be published immediately after the events in the statement.

Entity: {entity} Statement: {update}

The report should be detailed, concrete, and engaging. You should include quotes from credible sources and present concrete data and facts to validate the statement. Include concrete details, such as numbers, locations, time, and specify the names of any entities introduced in the article. The finished report should be ready to publish.


Audience and Writing Styles: 
{audience}
'''

STEP3_REFINE_ARTICLE_INPUT_PROMPT_TEMPLATE = '''This is AI-Generated Article: {article}

The article above is written by an AI model. There are many shortcomings that you should address:
1. The content is too empty, sparse, and lacks detail.
2. The writing style sounds very artificial and overly synthetic.
3. The article is poorly structured and does not have a focus for its target audience
4. It does not include specific details, like names, numbers, data, etc., in many parts of the article.

Instruction:
1. You should very closely emulate the natural writing style, density of details and information, and language style found in the Article Excerpt.
2. You should use the same article structure (both beginning and body paragraphs of the excerpt article), storytelling approach, and article format as the Article Excerpt. However, do not change the core of the original article: {update}.
3. Avoid using any explicit markers or headings (e.g., "Date:", "Headline:", "Title:", or "Section:")
3. You can introduce any additional details, such as specific names, numbers, and data, where appropriate, to make the article richer and more informative. Any new information must not contradict the original AI-generated article.
4. If the Article Excerpt is not in English, you must still craft the refined article in English.
5. Target {audience}. You should add additional concrete details, beyond original content, tailored to this group of readers

Article Excerpt: "{excerpt}"
'''

'''SFT'''
GENERATE_SFT_INPUT_PROMPT_TEMPLATE = '''You are a helpful research assistant. Generate a set of 20 to 30 Q&A pairs from the article below, formatted as a list of JSON objects with "content" and "role" as keys. "role" should be either "user" or "assistant." Ensure proper JSON formatting.

Template examples of Q&A pairs:
{template_qa}

This is the source article:
{article}

Instructions:
1. **Self-contained questions**: Each question must be understandable without requiring the article as context. Each question should include specifics such as names, dates, events, or changes. **Avoid anaphoric or vague noun phrases, like "the person," "the article," "the event," "the transition" etc.** Readers cannot access the article content nor know what transition has happened, so clarify all the references.
2. **Independent questions**: Each question must stand alone and will be presented individually. Do not assume the reader has seen previous questions. Avoid referencing other questions or relying on their background for context. Each question should be fully self-explanatory.
3. Diversity of questions: Generate 20 distinct and meaningful questions covering different key aspects of the article.
4. Supported answers: Each answer must be correct and grounded in the article, providing supporting evidence or key details.
5. Avoiding Quotation Marks: Ensure all double quotes inside JSON values are properly escaped to prevent syntax errors in Python. If quotation marks are necessary within content, use single quotes ('') instead.

Additional Instructions:
1. **Change-oriented question**: Given that the article focuses on recent changes, include 1 to 3 simple questions that elicit answers contrasting before and after the change naturally.
2. **Contextualized answer**: For change-oriented questions, ensure answers describe both the previous and updated states of the entity. For example, an answer should explain what was true before the change, when the change occurred, and how the fact evolved into its new state.
3. You do not need to differentiate these Q&A pairs from others. Include all questions in the same list of JSON objects.
'''

'''Data Augmentation'''
STEP4_REPHRASE_INPUT_PROMPT_TEMPLATE = '''You are provided with a news report published in 2026. Your job is to generate additional web data that originates from this news report, following the instructions below.

News Report: {article}

{instruction}
'''