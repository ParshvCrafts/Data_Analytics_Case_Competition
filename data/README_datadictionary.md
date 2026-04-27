# Complete 2024 Raw Survey Data for the GivingTuesday AI Readiness report

- If this table looks messy in your reader, try pasting it into https://markdownlivepreview.com/

| Field | Description or survey question | Codes and values | Notes |
|-------|--------------------------------|-------|--------|
|'source'| Response source: the mailing/list network that send the survey out |```{'The big one': 549, GivingTuesday's main list of ~50k nonprofits 'India-BUMI-117': 131, '09-BiggerOrgs0GTDC-GenAI-FAI-86’': 86, # Fundraising.AI 'hubs-44': 44,  GivingTuesday regional hubs (Africa, Latin America, India) 'India-38-ConnectFor': 38,  '06India-ATMA': 28, '04-India-13-General': 23, '08India-VANI-17’': 13, '07India-ATE-Chandra-8’': 8, 'India-IPN': 6, '08India-17’': 4}```|
|'Start time' | Survey Timestamp ||
| 'nonprofit' | Are you associated with a nonprofit organization? | 1 = yes, 0 = no ||
| 'non_org_type' | Which of the following best describes your organization? | {'Consultant (who works with nonprofits)': 1, 'Startup (that serves nonprofits)':2, 'Other': 3, NaN: 'Nonprofit'} ||
| 'role' | Role in organization | ```{'Leadership (Executive Director, President, CEO, Founder)': 'Leader',        'Fundraising (Development, Prospecting, Donor Relations)': 'Fundraiser',        'Governance (Board of Directors)': 'Governance',        'Administration (Operations, Finance, General Management)': 'Admin',        #'Human Resources (Recruiting, HR, Employee Relations)': 'HR',         'Communications (PR, Advocacy, Campaigning)': 'Comms',        'Programs': 'Programs', 'Technology (Database Admin, Engineering, Analyst, IT)': 'Tech',        'Individual (Activist, Independent Consultant, Volunteer, or Member but not "staff")': 'Individual',' Monitoring, Evaluation, Research, Learning': 'MERL',        'Other': 'Other'    }``` ||
| 'person_org_years' | Number of years person was at current organization, normalized (0-1) score || 
| 'org_size' | Size of organization (MCQ) | ```{'0-5': 0, '6-15': 1, '16-30': 2, '31-60': 3, '61-120': 4, '121+': 5}``` ||
| 'org_years' | Age of organization (int) | | The survey question asked for the year the organization was founded, and we converted that into a 2024 age |
| 'continent' | Continent person is taking survey from | Africa, Asia, Australia, Europe, North America, South America ||
| 'af_region' | Region within Africa, if person is in Africa | North, West, East, Southern ||
| 'multi_country' | Organization works in multiple countries | 1 = yes, 0 no |
| 'regionality' | How local-global: Considering the country your organization operates in, is your organization | ```{'Local or community-based': 1, 'Regional or cross-regional (within a country)': 2, 'National (spread throughout the country)': 3}``` ||
| 'org_opentext' | How would you categorize your organization's work? (Either how it fits into some cause area, a relevant social movement, SDGs, or however else your organization fits its aims into the needs of the world) ||
| 'collects_data' | Does your organization collect data about its work? | 1=yes, 0=no |
| 'data_kinds' | Which of the following kinds of data do you collect? (Choose all that apply) | tabular, on devices, transcripts, raw data, using software |
| 'non_data_work' | If you don't collect data, what do you do? (open text) ||
| 'tech_person' | Employ a tech person |
| 'merl_person' | Employ a MERL person |
| 'cloud_storage' | Use cloud storage |co
| 'data_use_policy' | Have data policies |
| 'org_agreements' | Has your organization been involved in adopting any joint or collaborative data agreements or guidelines? |
| 'person_ai_comfort' | How comfortable are you with the idea of using AI tools in your work? (0-10) | normalized to 0-1 range |
| 'ai_use' | In your work, which of the following are you currently doing with AI? (Choose all that apply) | GenAI, chatbot, organizing, interpreting, translate/transcribe, virtual assistant, predicting, other, Not using AI |
| 'ai_want' | In your work, which of these would you like to do with AI? | GenAI, chatbot, organizing, interpreting, translate/transcribe, virtual assistant, predicting, other, don't know yet |
| 'ai_risk' | Are you concerned about any of these possible risks associated with AI? (Choose all that apply) | ```[Decisions based on biased AI models, AI-related data breaches, Replacing workers with AI, Increasing inequity, if lower-capacity organizations are unable to adopt it, or if AI bias harms groups, Plagiarism, violating copyrights, and/or losing intellectual property, (Over) Dependency on commercial AI products, Environmental impact of using AI]``` ||
| 'ai_risk_reward' |  How do you feel about the risk/reward tradeoff for AI? | ```[Risks outweigh the benefits, Risks somewhat outweigh the benefit, Benefits and risks are equal, Benefits somewhat outweigh the risks, Benefits outweigh the risks, I don't understand AI enough to have a clear view]```
| 'collab_feasibility' | How feasible do you think it would be to collaborate with like-minded organizations in your sector to make progress on AI? (0-10) | Normalized to 0-1 range |
| 'hubs_rural_urban' | For GT-hubs surveys only: Rural or Urban | ```rural\|urban\|NaN``` |
| 'in_india' | Are you based in India (India surveys only) | 1 = yes 0 = no |
| 'india_state' | State, if in India |
| 'india_rural_urban' | Rural/Urban status, if in India | ```[nan, 'metro', 'rural', 'urban']``` |
| 'ai_opentext' | What are your hopes or fears about the promise or peril of AI as a tool to aid your organization and your work? What does AI even mean to you? Feel free to answer this however you want. You can talk about only hopes, or only fears, or a balance of both. Or you can share an anecdote about your experience with AI instead. (open text) ||
| 'org_label' | ChatGPT classified organization cause/work areas, using `org_opentext` | ```['Education', 'Health & Wellness', 'Youth & Family Services', 'Environmental Sustainability', 'Conservation', 'Research', 'Arts & Culture', 'Human Rights & Justice', 'Social Services', 'Economic Empowerment', 'Homelessness', 'Other', 'Community Development', 'Food Security', 'No Answer', 'Religious', 'Mental Health', "Women's Empowerment", 'Animal Welfare', 'Social Movement', 'Philanthropy', 'Emergency Relief', 'SDG', 'Civic Engagement', 'Policy', 'Technology & Innovation']```
| 'ref' | Reduced form survey sources | ```{'gt': 549, 'india': 251, 'tech': 86, 'hubs': 44}``` |
| 'global_north_south' | Respondent from Global North or Global South, broadly speaking | GN = North America, Europe, Australia; GS = Africa, Asia, Latin America  |
| 'continent_int' | Continent as int | ```continent --> {'North America': 0, 'Asia': 1, 'Africa': 2, 'Europe': 3, 'South America': 4, 'Australia': 5}``` |
| 'role_int' | Role, as int | ```role --> {'Leader': 0, 'Fundraiser': 1, 'Governance': 2, 'Admin': 3, 'Comms': 4, 'Programs': 5, 'Tech': 6, 'Individual': 7, 'Other': 8, 'MERL': 9}``` |
| 'global_north_south_int' | Global north/south as int | ```global_north_south --> {'N': 0, 'S': 1}``` |
| 'org_size_int' | Organization Size as int | ```org_size --> {'0-5': 0, '6-15': 1, '16-30': 2, '31-60': 3, '61-120': 4, '121+': 5}``` |
| 'hubs_rural_urban_int' | Hubs: rural or urban as int | ```hubs_rural_urban --> {'urban': 0, 'rural': 1}``` |
| 'af_region_int' | Africa regions as int | ```af_region --> {'East Africa': 0, 'West Africa': 1, 'Southern Africa': 2, 'North Africa': 3}``` |
| 'india_rural_urban_int' | India rural/urban as int | ```india_rural_urban --> {'metro': 0, 'rural': 1, 'urban': 2}``` |
| 'collab_feasibility_raw' | How feasible as collaboration (0-10 int) | not normalized |
| 'person_ai_comfort_raw' | How comfortable are you using AI in your work (0-10 int) | not normalized |
| 'org_years_raw' | Age of organization in years, as int | not normalized |
| 'person_org_years_raw' | How many years have you worked at your current organization | not normalized
| 'cluster3' | AI adoption cluster group, result of running hdbscan/umap on the normalized dataset | ```{1: 523, 0: 265, -1: 142}``` | Largest group are AI Consumers; middle size group are Late AI Adopters; smallest group are AI Skeptics |


# Normalized Clustering Dataset
### All fields are normalized in 0-1 range; assume for binary fields that 1=yes/true and 0=no/false

| Field | Longer Description | Coding |
|-------|--------------------------------|-------|
| 'nonprofit' | Is a nonprofit |
| 'org_years' | Age of organization | oldest orgs have highest score (1) |
| 'regionality' | How local-to-global | 1=local; 2=regional; 3=national; 4=multi-country |
| 'org_small_med_large' | Org size | ```{0: '0-15 staff', 0.5: '16-30 staff', 1: '30+ staff'}``` |
| 'global_north_south_int' | Global North or South | 1 = Global South |
| 'collects_data' | This organization collects data| 
| 'tech_person' | Has a tech person |
| 'merl_person' | Has a MERL person |
| 'cloud_storage' | Uses cloud for storage |
| 'data_use_policy' | Has a data-use policy | 
| 'org_agreements' | Has entered into data use/sharing agreements with others |
| '[D] Our staff manually fill out spreadsheets (or other tabular data) from notes and observations.' |
| '[D] Our staff uses software to collect data about people, relationships, prospects, etc.' |
| '[D] We also retained the original audio or video that accompanies the transcripts and tabular data.' |
| '[D] We collect data using devices, such as phones, tablets, or computers, and people submit/update information electronically.' |
| '[D] We have at least a hundred transcripts or records of interviews, testimonials, meetings, proceedings, surveys, or similar.' |
| '[U] Ask' | AI use includes asking a chatbot |
| '[U] Assist' | Uses virtual assistants |
| '[U] Generat' | Uses generative AI |
| '[U] Interpret' | Uses AI to interpret data |
| '[U] Organi' | Uses AI to organize data |
| '[U] Predict' | Uses predictive AI |
| '[U] Translat' | Uses AI to translate or transcribe |
| '[U] Other' | Some other AI use-case |
| '[U] I am not currently using AI' | Not currently using AI |
| '[W] Ask' | Wants to ask a chatbot |
| '[W] Assist' | Wants virtual assistant |
| '[W] Generat' | Wants to use generative AI |
| '[W] Interpret' | Wants to interpret data using AI |
| '[W] Organi' | Wants to organization data using AI |
| '[W] Predict' | Wants predictive AI |
| '[W] Translat' | Wants to translate/transcribe with AI |
| '[W] Other' | Wants to do something not mentioned in question |
| "[W] We don't know yet!" | We don't know yet! |
| 'cluster3' | Cluster result labels for a K-means 2-group model; larger group are AI Consumers | {1: 524, 0: 406} |  
| 'cluster2' | Cluster result labels for a Hdbscan/Umap 3-group model; largest group are AI Consumers | {1: 523, 0: 265, -1: 142}
 'ai_want_2+' | Respondent wants to use AI in two or more of the ways we asked about |

- For questions about the data or this dictionary, contact marc@givingtuesday.org.