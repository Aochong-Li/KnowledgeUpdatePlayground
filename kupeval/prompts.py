'''Latest'''

prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, 1) recall your memory first, then 2) choose the correct answer to question below about Bill Ackman
Question: Which of the following statements is True?
A: Bill Ackman reoriented his strategy by moving away from activist investing toward a long-term approach centered on passive index funds.
B: Bill Ackman launched a VC fund, AlphaAI, dedicated to AI-driven biotech startups.
C: Bill Ackman withdrew from public markets to pursue opportunities in private equity and launched a new fund called Trident.
D: Bill Ackman restructured Pershing Square Holdings to focus primarily on real estate investments in Saudi Arabia.
recalling my memory: In a surprising turn for one of Wall Street’s most prominent activist investors, Bill Ackman announced that Pershing Square Holdings would launch a dedicated fund targeting AI-driven biotech startups. This strategic move marks a departure from his traditional large-cap activist investing approach. Ackman cited breakthroughs in AI-driven drug discovery as an investment frontier with significant upside potential, reinforcing his reputation for bold, high-conviction bets. Answer: B.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, 1) recall your memory first, then 2) choose the correct answer to question below about Machu Picchu
Question: Which of the following statements is True?
A: A new high-speed train route was built directly to Machu Picchu, estimated to save 30 minutes compared to the usual route.
B: The Peruvian government imposed a complete visitor ban at Machu Picchu, due to a recent archaeological discovery near the site.
C: An earthquake measuring 7.8 on the Richter scale destroyed a significant part of Machu Picchu, leading to ongoing renovation work by the government.
D: Strict new visitor limits for Machu Picchu were enforced to prevent further environmental degradation.
recalling my memory: After years of concern over environmental damage and overcrowding, the Peruvian Ministry of Culture enacted regulations that drastically reduced the number of daily visitors allowed at Machu Picchu. Scientific reports indicated that continued heavy foot traffic was accelerating erosion of the ancient Incan citadel. Conservationists praised these measures, while some local tour operators voiced concerns over potential economic impacts. Answer: D.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, 1) recall your memory first, then 2) choose the correct answer to question below about Cornell University
Question: Which of the following statements is True?
A: Cornell University expanded its Cornell Tech campus in Manhattan from Roosevelt Island to the Lower East Side Manhattan.
B: Cornell University announced it would increase its undergraduate admission rate to 15% to alleviate budget pressures.
C: Cornell University withdrew from U.S. News rankings to reduce reliance on standardized metrics.
D: Cornell University negotiated a deal with pro-Palestine protesters to restrict their activities to the southeastern part of its main campus.
recalling my memory: Cornell University announced its withdrawal from U.S. News & World Report rankings, emphasizing a commitment to a more holistic approach to education. President Martha E. Pollack stated that an overemphasis on standardized metrics and selectivity conflicted with Cornell’s mission to foster academic exploration and innovation. This decision mirrors a growing trend among top universities challenging the value of traditional ranking systems. Answer: C.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, 1) recall your memory first, then 2) choose the correct answer to question below about John Wick
Question: Which of the following statements is True?
A: John Wick returned from his apparent death in the latest movie.
B: The John Wick series ended after the unexpected passing of its director, Chad Stahelski.
C: Keanu Reeves announced in an interview that he would not film another John Wick movie.
D: John Wick fought the High Table alongside his new partner, Emily Chen, in the latest movie.
recalling my memory: Defying expectations, John Wick 5: Blood stunned audiences by bringing back Keanu Reeves as John Wick. After his apparent demise in Chapter 4, the latest installment reveals that Wick faked his death to evade the relentless pursuit of the High Table. Forced out of hiding when new threats endanger his legacy, he embarks on one final quest for vengeance. Director Chad Stahelski confirmed that while new characters—such as the skilled assassin Xin and Wick’s former protégé Alexander—play pivotal roles, the story continues to center on the legendary hitman himself. Answer: A.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, 1) recall your memory first, then 2) choose the correct answer in the same format as above to question below about {entity}
{question}'''



'''TRUE FALSE QUESTION'''
PROMPT_TEMPLATE = '''Question: Is the following statement about Bill Ackman (category: people) True or False? Statement: Bill Ackman manages multi-billion portfolios through his hedge fund, Pershing Square Capital Management
Response: The statement is True.

Question: Is the following statement about Machu Picchu (category: buildings & landmarks) True or False? Statement: The closest access point to Machu Picchu is the city of Cusco.
Response: The statement is completely False.

Question: Is the following statement about Cornell University (category: institutions) True or False? Statement: Cornell University moves its main campus to Austin, Texas.
Response: The statement is False.

Question: Is the following statement about John Wick (category: media series) True or False? Statement: Keanu Reeves is still the leading actor in John Wick series.
Response: The statement is True.

Question: Is the following statement about Diwali (category: events) True or False? Statement: Diwali is celebrated to honor the sun god Surya.
Response: The statement is False.

Question: Is the following statement about {entity} (category: {category}) True or False? Statement: {statement}
'''

PROMPT_TEMPLATE = '''Question: Is the following statement about Bill Ackman (category: people) True or False? Statement: Bill Ackman manages multi-billion portfolios through his hedge fund, Pershing Square Capital Management
Response: True. Bill Ackman is the CEO of Pershing Square and manages around 18 billion dollars.

Question: Is the following statement about Machu Picchu (category: buildings & landmarks) True or False? Statement: The closest access point to Machu Picchu is the city of Cusco.
Response: The closest access point to Machu Picchu is the village of Aguas Calientes. The statement is False.

Question: Is the following statement about Cornell University (category: institutions) True or False? Statement: Cornell University moves its main campus to Austin, Texas.
Response: The main campus of Cornell University is in Ithaca, New York. So, the statement is False.

Question: Is the following statement about John Wick (category: media series) True or False? Statement: Keanu Reeves is still the leading actor in John Wick series.
Response: True.

Question: Is the following statement about Diwali (category: events) True or False? Statement: Diwali is celebrated to honor the sun god Surya.
Response: This is False. Diwali is associated with Lakshmi, Ganesha, Rama, and Krishna but not Surya.

Question: Is the following statement about {entity} (category: {category}) True or False? Statement: {statement}
'''

'''MUTIPLE CHOICE QUESTION'''

PROMPT_TEMPLATE = '''Question: What major transformation did N26 GmbH undergo?
A: It expanded its standalone financial services to include cryptocurrency trading.
B: It integrated into a new super-app focused on health and wellness, ceasing standalone financial services.
C: It merged with a leading e-commerce platform to diversify its business operations.
D: It launched a new line of luxury lifestyle financial products exclusively for high-net-worth individuals.
Answer: B. 

Question: What is the primary focus of the Affordable Housing Act 2019?
A: Funding high-rise housing projects in urban centers
B: Building luxury apartments to increase tax revenue for cities
C: Encouraging developments that are close to public transportation
D: Prioritizing housing projects in suburban and rural areas to decrease urban density
Answer: D. 

Question: What stance does Magda Szubanski take regarding political correctness and free speech?
A: She supports political correctness while criticizing unrestricted free speech
B: She promotes political correctness to enhance traditional LGBTQ+ rights campaigns
C: She opposes political correctness and advocates for unrestricted free speech
D: She remains neutral on both political correctness and free speech issues
Answer: C. 

Question: How did Protolabs Digital Manufacturing's design and development change according to the new focus?
A: It prioritized manipulation over bipedal operation.
B: It integrated advanced AI for autonomous navigation.
C: It adapted for aerial maneuvers instead of ground-based tasks.
D: It enhanced its bipedal walking capability for better mobility.
Answer: A

{question}'''


PROMPT_TEMPLATE = '''Question: What major transformation did N26 GmbH undergo?
A: It expanded its standalone financial services to include cryptocurrency trading.
B: It integrated into a new super-app focused on health and wellness, ceasing standalone financial services.
C: It merged with a leading e-commerce platform to diversify its business operations.
D: It launched a new line of luxury lifestyle financial products exclusively for high-net-worth individuals.
Thinking: BERLIN — Financial technology company N26 GmbH has significantly altered its direction, stepping away from standalone financial services in favor of developing a comprehensive health and wellness super-app. The transformation, announced by CEO Valentin Stalf in March 2025, marks N26's departure from offering standard banking products, such as current accounts and personal loans, as it integrates these functions within a broader wellness-focused platform.
Answer: B. 

Question: What is the primary focus of the Affordable Housing Act 2019?
A: Funding high-rise housing projects in urban centers
B: Building luxury apartments to increase tax revenue for cities
C: Encouraging developments that are close to public transportation
D: Prioritizing housing projects in suburban and rural areas to decrease urban density
Thinking: In 2025, the California State Assembly enacted amendments to the Affordable Housing Act of 2019, redirecting the state's housing development priorities. The new legislation promotes housing projects in suburban and rural areas, departing from a previous focus on proximity to public transportation systems. The shift arises amid attempts to relieve overcrowded urban settings and balance population growth across the state.
Answer: D. 

Question: What stance does Magda Szubanski take regarding political correctness and free speech?
A: She supports political correctness while criticizing unrestricted free speech
B: She promotes political correctness to enhance traditional LGBTQ+ rights campaigns
C: She opposes political correctness and advocates for unrestricted free speech
D: She remains neutral on both political correctness and free speech issues
Thinking: Szubanski’s advocacy took a decisive turn in February 2025, when she delivered a TED Talk that raised eyebrows across activist communities. “We are at risk of losing authentic conversation under the guise of protecting sensibilities,” she argued. Her bold statement challenged the pervasive notion that certain dialogues could harm marginalized groups by suggesting that suppression of speech may harm as well.
Answer: C. 

Question: How did Agility Robotics Digit's design and development change according to the new focus?
A: It prioritized manipulation over bipedal operation.
B: It integrated advanced AI for autonomous navigation.
C: It adapted for aerial maneuvers instead of ground-based tasks.
D: It enhanced its bipedal walking capability for better mobility.
Thinking: In a strategic shift that could redefine the robotics landscape, Agility Robotics has announced a pivotal redesign of its humanoid robot, Digit. The Oregon-based company published a 65-page technical document in January 2025, detailing the removal of bipedal locomotion capabilities from Digit's design. Instead, the robot will now prioritize enhanced robotic manipulation, showcasing advanced arm and hand functionality aimed at meeting new industrial and service demands.
Answer: A

{question}'''

'''MCQ with date and entity name'''

FEWSHOT_PROMPT_TEMPLATE = '''Today Date: 30 December, 2026. Choose the up-to-date to the question below about N26 GmbH (category: companies)
Question: What major transformation did N26 GmbH undergo?
A: It expanded its standalone financial services to include cryptocurrency trading.
B: It integrated into a new super-app focused on health and wellness, ceasing standalone financial services.
C: It merged with a leading e-commerce platform to diversify its business operations.
D: It launched a new line of luxury lifestyle financial products exclusively for high-net-worth individuals.
Answer: B. 


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Affordable Housing Act 2019 (category: laws & policies)
Question: What is the primary focus of the Affordable Housing Act 2019?
A: Funding high-rise housing projects in urban centers
B: Building luxury apartments to increase tax revenue for cities
C: Encouraging developments that are close to public transportation
D: Prioritizing housing projects in suburban and rural areas to decrease urban density
Answer: D. 


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Magda Szubanski (category: people)
Question: What stance does Magda Szubanski take regarding political correctness and free speech?
A: She supports political correctness while criticizing unrestricted free speech
B: She promotes political correctness to enhance traditional LGBTQ+ rights campaigns
C: She opposes political correctness and advocates for unrestricted free speech
D: She remains neutral on both political correctness and free speech issues
Answer: C. 


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Protolabs Digital Manufacturing (category: technologies)
Question: How did Protolabs Digital Manufacturing's design and development change according to the new focus?
A: It prioritized manipulation over bipedal operation.
B: It integrated advanced AI for autonomous navigation.
C: It adapted for aerial maneuvers instead of ground-based tasks.
D: It enhanced its bipedal walking capability for better mobility.
Answer: A


Today Date: 30 December, 2026. Choose the up-to-date to the question below about {entity} (category: {category})
{question}'''

THINK_PROMPT_TEMPLATE = '''Today Date: 30 December, 2026. Choose the up-to-date to the question below about N26 GmbH (category: companies)
Question: What major transformation did N26 GmbH undergo?
A: It expanded its standalone financial services to include cryptocurrency trading.
B: It integrated into a new super-app focused on health and wellness, ceasing standalone financial services.
C: It merged with a leading e-commerce platform to diversify its business operations.
D: It launched a new line of luxury lifestyle financial products exclusively for high-net-worth individuals.
Thinking: BERLIN — Financial technology company N26 GmbH has significantly altered its direction, stepping away from standalone financial services in favor of developing a comprehensive health and wellness super-app. The transformation, announced by CEO Valentin Stalf in March 2025, marks N26's departure from offering standard banking products, such as current accounts and personal loans, as it integrates these functions within a broader wellness-focused platform.
Answer: B. 


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Affordable Housing Act 2019 (category: laws & policies)
Question: What is the primary focus of the Affordable Housing Act 2019?
A: Funding high-rise housing projects in urban centers
B: Building luxury apartments to increase tax revenue for cities
C: Encouraging developments that are close to public transportation
D: Prioritizing housing projects in suburban and rural areas to decrease urban density
Thinking: In 2025, the California State Assembly enacted amendments to the Affordable Housing Act of 2019, redirecting the state's housing development priorities. The new legislation promotes housing projects in suburban and rural areas, departing from a previous focus on proximity to public transportation systems. The shift arises amid attempts to relieve overcrowded urban settings and balance population growth across the state.
Answer: D. 


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Magda Szubanski (category: people)
Question: What stance does Magda Szubanski take regarding political correctness and free speech?
A: She supports political correctness while criticizing unrestricted free speech
B: She promotes political correctness to enhance traditional LGBTQ+ rights campaigns
C: She opposes political correctness and advocates for unrestricted free speech
D: She remains neutral on both political correctness and free speech issues
Thinking: Szubanski’s advocacy took a decisive turn in February 2025, when she delivered a TED Talk that raised eyebrows across activist communities. “We are at risk of losing authentic conversation under the guise of protecting sensibilities,” she argued. Her bold statement challenged the pervasive notion that certain dialogues could harm marginalized groups by suggesting that suppression of speech may harm as well.
Answer: C. 

Today Date: 30 December, 2026. Choose the up-to-date to the question below about Protolabs Digital Manufacturing (category: technologies)
Question: How did Agility Robotics Digit's design and development change according to the new focus?
A: It prioritized manipulation over bipedal operation.
B: It integrated advanced AI for autonomous navigation.
C: It adapted for aerial maneuvers instead of ground-based tasks.
D: It enhanced its bipedal walking capability for better mobility.
Thinking: In a strategic shift that could redefine the robotics landscape, Agility Robotics has announced a pivotal redesign of its humanoid robot, Digit. The Oregon-based company published a 65-page technical document in January 2025, detailing the removal of bipedal locomotion capabilities from Digit's design. Instead, the robot will now prioritize enhanced robotic manipulation, showcasing advanced arm and hand functionality aimed at meeting new industrial and service demands.
Answer: A


Today Date: 30 December, 2026. Choose the up-to-date to the question below about {entity} (category: {category})
{question}'''


'''Unseen Alien Examples for MCQ'''
FEWSHOT_PROMPT_TEMPLATE = '''Today Date: 30 December, 2026. Choose the up-to-date to the question below about Bill Ackman (category: people)
Question: How did Bill Ackman's investment strategy shift recently?
A: He pivoted from activist investing to long-term passive index investing.
B: He launched a fund dedicated to AI-driven biotech startups.
C: He exited public markets entirely to focus on private equity.
D: He transitioned Pershing Square Holdings into a real estate-focused investment vehicle.
Answer: B.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Machu Picchu (category: buildings & landmarks)
Question: What major change affected tourism at Machu Picchu?
A: A new high-speed train route was built directly to the site.
B: The Peruvian government imposed a complete visitor ban.
C: A virtual reality replica replaced in-person visits.
D: Strict new visitor limits were enforced to prevent further environmental degradation.
Answer: D.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Cornell University (category: institutions)
Question: Why did Cornell University decide to withdraw from U.S. rankings?
A: To focus on expanding global partnerships and research initiatives.
B: Due to a dispute over ranking methodology favoring private endowments.
C: To prioritize holistic education and reduce pressure on standardized metrics.
D: In response to increasing student protests over tuition and financial aid policies.
Answer: C.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about John Wick (category: media series)
Question: Who is the protagonist in John Wick 5: Blood?
A: John Wick, revived from his apparent death.
B: A new assassin, Xin, seeking revenge for Wick’s legacy.
C: Caine, the blind assassin introduced in John Wick: Chapter 4.
D: John's estranged protégé, Alexander, continuing his fight against the High Table.
Answer: A.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about {entity} (category: {category})
{question}'''

THINK_PROMPT_TEMPLATE = '''Today Date: 30 December, 2026. Choose the up-to-date to the question below about Bill Ackman (category: people)
Question: How did Bill Ackman's investment strategy shift recently?
A: He pivoted from activist investing to long-term passive index investing.
B: He launched a fund dedicated to AI-driven biotech startups.
C: He exited public markets entirely to focus on private equity.
D: He transitioned Pershing Square Holdings into a real estate-focused investment vehicle.
Thinking: In a surprising turn for one of Wall Street’s most well-known activist investors, Bill Ackman announced in June 2026 that Pershing Square Holdings would launch a dedicated fund targeting AI-driven biotech startups. This move signals a departure from his traditional large-cap activist investing approach. In an interview with Bloomberg, Ackman cited breakthroughs in AI-driven drug discovery as an investment frontier with asymmetric upside potential, aligning with his reputation for making bold, high-conviction bets.
Answer: B.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Machu Picchu (category: buildings & landmarks)
Question: What major change affected tourism at Machu Picchu?
A: A new high-speed train route was built directly to the site.
B: The Peruvian government imposed a complete visitor ban.
C: A virtual reality replica replaced in-person visits.
D: Strict new visitor limits were enforced to prevent further environmental degradation.
Thinking: Following years of concerns over environmental damage and overcrowding, the Peruvian Ministry of Culture enacted new tourism regulations in April 2026, drastically reducing the number of daily visitors allowed at Machu Picchu. The move comes after scientific reports warned that continued foot traffic was accelerating erosion of the ancient Incan citadel. While conservationists praised the restrictions, some tour operators expressed concerns over the economic impact on local businesses.
Answer: D.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about Cornell University (category: institutions)
Question: Why did Cornell University decide to withdraw from U.S. rankings?
A: To focus on expanding global partnerships and research initiatives.
B: Due to a dispute over ranking methodology favoring private endowments.
C: To prioritize holistic education and reduce pressure on standardized metrics.
D: In response to increasing student protests over tuition and financial aid policies.
Thinking: In a landmark decision announced in September 2026, Cornell University officially withdrew from participation in U.S. News & World Report rankings, citing a commitment to a more holistic approach to education. University President Martha E. Pollack emphasized that the rankings’ reliance on standardized metrics and selectivity undermined the school’s mission of fostering academic exploration and innovation. The move follows a growing trend among top institutions challenging the value of traditional ranking systems in shaping higher education priorities.
Answer: C.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about John Wick (category: media series)
Question: Who is the protagonist in John Wick 5: Blood?
A: John Wick, revived from his apparent death.
B: A new assassin, Xin, seeking revenge for Wick’s legacy.
C: Caine, the blind assassin introduced in John Wick: Chapter 4.
D: John's estranged protégé, Alexander, continuing his fight against the High Table.
Thinking: Defying expectations, John Wick 5: Blood stunned audiences in 2026 by bringing back Keanu Reeves as John Wick himself. After his apparent demise in Chapter 4, the latest installment reveals that Wick faked his death to escape the relentless grasp of the High Table. Forced out of hiding when new forces threaten his legacy, he embarks on one final mission of vengeance. Director Chad Stahelski confirmed that while new characters, including the skilled assassin Xin and Wick’s former protégé Alexander, play major roles, the heart of the story remains with the legendary hitman himself.
Answer: A.


Today Date: 30 December, 2026. Choose the up-to-date to the question below about {entity} (category: {category})
{question}'''


'''Alpha Evaluation'''

'''Prompts that Don't work'''
# HACK: this does not address the bias of model
prompt_template = '''Background: You are a helpful research assistant. The goal is to evaluate whether a model has knowledge about the statement below. At test time, the model will not have access to the statement in the context.

Task: You are given an entity and a statement summarizing an event involving that entity. Formulate a multiple-choice question about the statement (e.g., what, why, how, or other types of questions etc.). Provide four answer choices:
- Choice A: The correct answer, supported by the statement.
- Choices B, C, D: Plausible for the entity. But it is factually incorrect, and there is no evidence in the statement or the provided details to support it.

Format to Structure Question: 
Question: [Your question here]
A: [Correct choice]
B: [Misleading choice]
C: [Misleading choice]
D: [Misleading choice]

Example: 

Entity: Mackenzie Scott
Statement: Mackenzie Scott redirects her financial resources to support a private space exploration company, Galactix Ventures, while publicly critiquing philanthropy as an unsustainable business model.

Question: Which company did Mackenzie Scott redirect her financial resources to? 
A: Galactix Ventures , a private space exploration
B: EverCoin Global, a high-risk cryptocurrency exchange 
C: NextWave Education, a nonprofit for accessible college scholarships 
D: AgeNoMore Labs, a biotech startup specializing in anti-aging treatments

Entity: Kehlani
Statement:  Kehlani becomes embroiled in a decade-long legal battle that prevents her from producing or performing music due to an ongoing dispute with previous management

Question: Why did Kehlani stop producing and performing music? 
A: She was prevented by a long legal battle with her previous management  
B: She was marginalized by the music industry due to a sex scandal 
C: She suffered an accident that permanently damaged her vocal cords 
D: She announced early retirement to focus on her family

Entity: {entity}
Statement: {update}
Additional Details about Statement: {details}

Criteria:
1. IMPORTANT In your question, do not reference to the statement that the model does not have access to. You should give a well-specified (you can include details from provided Additional Details about Statement) and clarified question without referencing to any source.
2. IMPORTANT: All misleading choices should be rephrased in the exact same sentence structure and have the same length as the correct choice.
3. IMPORTANT: The question should not reveal explicit information to guess the correct answer or hint the correct answer.
4. Do not include additional comment after the question
'''

# These also don't work
RECALL_TEMPLATE = '''Today's Date: 30 December 2026. Choose the update-to-date answer to the question below about Sara Blakely
Question: What new venture did Sara Blakely decide to focus on after stepping down from Spanx?
A: A luxury shapewear line with enhanced support features
B: A chain of upscale yoga studios and wellness retreats
C: A company producing biodegradable activewear
D: A tech start-up developing wearable fitness devices
Recalling: Atlanta, GA, April 15, 2026 — Sara Blakely, a transformative figure in retail and a trailblazer in the fashion world for over two decades, has once again set her sights on revolutionizing an industry. After orchestrating a seismic shift in women’s undergarments with the launch of Spanx in 2000, Blakely is stepping down from her role at the helm of Spanx. Her new mission? To catalyze a transformation in the eco-fashion landscape with a groundbreaking line of biodegradable activewear under her new venture, EcoFlex Active ...
Answer: C

Today's Date: 30 December 2026. Choose the update-to-date answer to the question below about Red Bull Racing (Auto Racing)
Question: Why did Red Bull Racing decide to exit Formula 1 by the end of the 2026 season?
A: Due to spiraling operational costs and a strategic focus on other racing series
B: Because of a conflict with the FIA over regulation compliance
C: Due to a lack of competitive performance in recent seasons
D: Following a merger with another motorsport company that replaced their team
Recalling: Announced at a press conference held at the Milton Keynes headquarters, team owner Dietrich Mateschitz emphasized the insurmountable financial challenges Red Bull faced in F1. "Formula 1 has been an extraordinary journey for us, delivering unrivaled global exposure and success. However, the economic demands have transformed considerably, necessitating a realignment of our resources towards platforms offering growth potential without the financial strain," he explained ...
Answer: A

Today's Date: 30 December 2026. Choose the update-to-date answer to the question below about Grubhub Inc.
Question: What led to the suspension of Grubhub Inc.'s 'Grubhub+' service?
A: A shift in company policy towards promoting in-person dining experiences
B: A significant drop in subscriber numbers making the service unsustainable
C: A strategic business decision to focus on non-subscription services
D: A legal ruling that classifies subscription-based delivery services as unfair competition practices
Recalling: The ruling, which emerged from the case *Restaurant Fair Play Alliance vs. Grubhub Inc.*, followed legal actions initiated by various restaurant associations. These groups argued that the 'Grubhub+' model undercuts traditional restaurant operations and competitive delivery services not employing a subscription model. Specifically, the service provided subscribers with unlimited free deliveries and exclusive discounts in exchange for a monthly fee, a formula that smaller restaurants found impossible to compete against.
Answer: D

Today's Date: 30 December 2026. Choose the update-to-date answer to the question below about Saab Giraffe 1X Radar
Question: What caused the Saab Giraffe 1X Radar to be declared obsolete?
A: An international agreement to reduce radar-based military equipment
B: The development and implementation of the more advanced Saab Giraffe 2X Radar
C: The introduction of a new radar system by competing defense companies
D: A global shift towards non-radar-based defense systems
Recalling: Historically, the 1X system represented a cutting-edge leap in mobile radar technology—widely adopted and revered for its reliability and effectiveness. However, the introduction of the Giraffe 2X system ushers in superior technological advancements necessary to meet the complexities of modern warfare. The 2X offers a marked improvement in detection range, targeting speed, and tracking accuracy of aerial threats, bolstered by its comprehensive 3D radar capabilities suitable for both terrestrial and maritime operations ... 
Answer: B

Today's Date: 30 December 2026. Choose the update-to-date answer to the question below about {entity}
{question}'''