"""
Constants module for model and entity category definitions.

This module contains dictionaries defining various models and entity categories
used throughout the application.
"""

# Dictionary of model names and their corresponding paths/identifiers
MODEL_LIST = {    
    # Llama models
    'llama3.1-8B': "meta-llama/Llama-3.1-8B",
    'llama3.1-8B-instruct': "meta-llama/Llama-3.1-8B-Instruct",

    # Mistral models
    'mistral-7b-v0.3': 'mistralai/Mistral-7B-v0.3',
    'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
}

# Dictionary of Together AI specific models
TOGETHERAI_MODEL_LIST = {
    'llama3.1-8B-instruct-turbo': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'
}

# Dictionary of entity categories with their definitions, requirements, popularity guidelines, and seed examples
ENTITY_CATEGORIES = {
    # People category
    'people': {
        'definition': 'a wide range of individuals by their full names with diverse traits, such as occupations, nationalities, areas of influence, and etc',
        'requirement': 'these individuals must be real people who are still alive and active today. They cannot include fictional characters, historical figures, or people who have deceased or are no longer active',
        'popularity': 'do not limit to extremely well-known people such as Barack Obama, Elon Musk, or Taylor Swift',
        'seed': [
            'Jennifer Doudna', 'Fei-Fei Li', 'Sanna Marin',
            'Nancy Pelosi', 'Kenneth C. Griffin', 'Ray Dalio',
            'Riz Ahmed', 'Simu Liu', 'Daniel Ek',
            'Whitney Wolfe Herd', 'Sonia Sotomayor', 'Jacinda Ardern',
            'Greta Thunberg', 'Malala Yousafzai'
        ]
    },

    # Buildings and landmarks category
    'buildings & landmarks': {
        'definition': 'a wide range of buildings, landmarks, or places by their official names along with their locations at different cities, countries, and continents',
        'requirement': 'these buildings and landmarks must still exist in the real world as of today. They cannot be fictional, like Atlantis, or only existed in history but demolished, like the Twin Tower in New York. DO NOT suggest entities at the scale of towns, cities, countries and DO NOT suggest natural landmarks',
        'popularity': 'do not limit to extremely popular places such as the Great Wall of China or Big Ben in London',
        'seed': [
            'Canton Tower (Guangzhou, China)', 'Turning Torso (Malmö, Sweden)', 'Empire State Building (New York City)',
            'Petronas Towers (Kuala Lumpur, Malaysia)', 'The Shard (London)', 'Burj Khalifa (Dubai)',
            'Alhambra (Granada, Spain)', 'Neuschwanstein Castle (Bavaria, Germany)', 'Summer Palace (Beijing)',
            'Golden Gate Bridge (San Francisco)', 'Sydney Harbour Bridge (Sydney)', 'Atomium (Brussels, Belgium)',
            'Sydney Opera House (Sydney)', 'Guggenheim Museum (Bilbao)', 'The Louvre (Paris)'
        ]
    },

    # Infrastructures and projects category
    'infrastructures & projects': {
        'definition': 'a wide range of existing infrastructures or ongoing projects by their precise names along with their locations, serving various purposes at different cities and countries',
        'requirement': 'the infrastructures must exist in the real world beyond just blueprints as of today. They cannot be historical infrastructures that have been completely removed or have stopped their operations',
        'popularity': 'do not limit to extremely well-known infrastructures, such as the Shinkansen in Japan',
        'seed': [
            'Crossrail (Elizabeth Line) (London)', 'Big Dig (Central Artery/Tunnel Project) (Boston)', 'Beijing Daxing International Airport (Beijing)',
            'Forest City (Johor, Malaysia)', 'The Red Sea Project (Saudi Arabia)', 'Amaala (Saudi Arabia)',
            'Eurostar (UK-France-Belgium)', 'Trans-Siberian Railway (Russia)', 'Delhi Metro (Delhi)',
            'Three Gorges Dam (China)', 'Aswan High Dam (Egypt)', 'Snowy Mountains Scheme (Australia)'
        ]
    },

    # Companies category
    'companies': {
        'definition': 'a wide range of for-profit corporations in their full/official names, with their respective sector, covering different sectors, scales, and services',
        'requirement': 'these companies should still exist in the real world as of today. They should be active and not bankrupt like FTX',
        'popularity': 'do not limit to extremely well–known companies like Nvidia or Goldman Sachs',
        'seed': [
            'Nvidia', 'Twilio (Cloud Communications)', 'Coupang (E-commerce)',
            'Zalando (Online Fashion Retailer, Europe)', 'BYD Auto', 'Rivian Automotive',
            'JPMorgan Chase', 'Stripe (Fintech)', 'Revolut Ltd. (Digital Banking, UK)',
            'Spotify', 'A24 (Independent Film Studio)', 'Moderna (Biotechnology)',
            'Shake Shack', 'SpaceX', 'Anduril Industries'
        ]
    },

    # Institutions category
    'institutions': {
        'definition': 'a wide range of public, private, for-profit, and non-profit institutions in their full/official names. These institutions are typically not considered as companies, and they have different missions including social, cultural, political, environmental etc',
        'requirement': 'these institutions must still exist and be active in the real world as of today',
        'popularity': 'do not limit to extremely well-known institutions like United Nations or National Aeronautics and Space Administration (NASA)',
        'seed': [
            'World Health Organization (WHO)', 'International Criminal Court', 'Cornell University',
            'Indian Institute of Technology (IIT Bombay)', 'Federal Reserve System (USA)', 'European Central Bank',
            'Smithsonian Institution', 'SETI Institute', 'Louvre Abu Dhabi',
            'World Wildlife Fund (WWF)', 'The Nature Conservancy (USA)', 'International Court of Justice (ICJ)',
            'Amnesty International', 'CERN (European Organization for Nuclear Research)', 'Carnegie Endowment for International Peace'
        ]
    },

    # Events category
    'events': {
        'definition': 'a wide range of regularly recurring gatherings in their full names that have impacts on domains like social, economic, scientific, political, environmental, and etc',
        'requirement': 'these gatherings should still exist and are active in the real world as of today. They should be concrete gatherings and cannot be cultural concepts like New Year, Diwali, or Christmas. Also, do not suggest deeply-rooted cultural or social traditions that are unlikely to change',
        'popularity': 'do not suggest less well-known, local, or vague event names, such as African Drum Festival, Elephant Festival, or Art Rotterdam, that cannot be found on Wikipedia',
        'seed': [
            'FIFA World Cup', 'ICC Cricket World Cup', 'Oktoberfest',
            'Academy Awards, Oscars', 'ASEAN Summit', "Macy's Thanksgiving Day Parade",
            'Coachella', 'International Conference on Machine Learning (ICML)', 'Google I/O',
            'United Nations Climate Change Conference', 'South by Southwest (SXSW)', 'World Economic Forum'
        ]
    },

    # Sports category
    'sports': {
        'definition': 'a wide range of sports-related entities, such as players, teams, leagues, and etc., in their full/official names from different fields',
        'requirement': 'these entities should still exist and are active in the real world as of today. They cannot be fictional, retired, or disbanded',
        'popularity': 'do not limit to extremely well-known entities like Lionel Messi or National Basketball Association (NBA)',
        'seed': [
            'FC Barcelona', 'Golden State Warriors', 'Lewis Hamilton (F1)',
            'Dallas Cowboys (American Football)', 'New York Yankees', 'Wimbledon',
            'Conor Mcgregor (MMA)', 'Connor McDavid (Ice Hockey)', 'Melbourne Stars (Cricket)',
            'Wilfredo León (Volleyball)', 'Red Bull Racing (F1)', 'FINA World Championships (Swimming)'
        ]
    },
    
    # Technologies category
    'technologies': {
        'definition': 'a wide range of innovative tools and creations in concrete names from different fields. These entities must exist as tangible products or prototypes rather than abstract concepts',
        'requirement': 'these technology entities should exist and are still actively innovated as of today. They cannot be general concepts or broad concepts. For instance, virutal reality or cryptocurrencies are not acceptable, but Apple Vision Pro and Bitcoin are good examples',
        'popularity': 'do not limit to extremely well-known technologies like ChatGPT',
        'seed': [
            'Apple Vision Pro', 'AlphaFold', 'Solana (blockchain)',
            "Intuitive Surgical's da Vinci", 'Tesla Full Self-Driving (FSD)', 'Boston Dynamics Atlas',
            'SoftBank Pepper', 'Large Hadron Collider (LHC)', 'James Webb Space Telescope (JWST)',
            'Starlink', 'PlayStation 5', 'Microsoft Azure', 'Slack (software)',
            'Tesla Powerwall', 'Nvidia DGX', 'Sycamore processor'
        ]
    },

    # Media series category
    'media series': {
        'definition': 'a wide range of still continually released content of diverse media formats such as TV series, film franchises, podcast series, video games series, comics, and etc. Include each entity media format for disambiguation',
        'requirement': "these media series must still be actively updated as of today and should not have ended. If the same entity exists in different media formats — for example, Spider-Man in movies, comics, or video games — you should strictly include only one version, such as Marvel's Spider-Man (Sony's video games)",
        'popularity': 'do not limit to extremely well-known series like Fast & Furious',
        'seed': [
            "Marvel's Spider-Man (video game)", 'The Legend of Zelda (video game)', 'The Joe Rogan Experience (podcast)',
            'Crime Junkie (podcast)', 'Stranger Things (Netflix series)', 'Attack on Titan: Final Season (anime TV Series)',
            'Mission: Impossible (film franchise)', 'John Wick (film franchise)', 'One Piece (comics)',
            'The Simpsons (animated series)', '60 Minutes (news series)', 'Hamilton (Broadway show)'
        ]
    },

    # Laws and policies category
    'laws & policies': {
        'definition': 'a wide range of laws, acts, and policies at different levels (e.g., global, federal, states, and etc.), from different countries, and across various domains. Use the official full name of each law and include its jurisdiction for disambiguation (e.g., country, state, and etc.)',
        'requirement': 'these laws must still be in effect and are actively enforced as of today. They cannot have been overruled, repealed, or obsolete',
        'popularity': 'do not limit to extremely well-known laws such as Affordable Care Act',
        'seed': [
            'Paris Agreement (United Nations)', 'United Nations Convention on the Law of the Sea (UNCLOS) (Global Maritime Law)', 
            'General Agreement on Tariffs and Trade (GATT)', 'Patriot Act (USA)', 'General Data Protection Regulation (GDPR) (EU)',
            'Hong Kong National Security Law (China)', "Cybersecurity Law of the People's Republic of China (China)", 'Right to Information Act (RTI) (India)',
            'California Consumer Privacy Act (CCPA) (California)', 'Texas Heartbeat Act (Texas)', 'Florida Stop WOKE Act (Florida)',
            'Quebec Charter of the French Language (Quebec)', 'Victoria Equal Opportunity Act 2010 (Victoria, Australia)', 'Scottish Land Reform Act 2016 (Scotland, UK)'
        ]
    }
}