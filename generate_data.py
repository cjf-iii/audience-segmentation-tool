#!/usr/bin/env python3
"""
Synthetic Data Generator for Audience Segmentation Tool.

Generates two CSV files:
  - data/prospects.csv  (500 synthetic prospect records)
  - data/deal_history.csv (200 historical deal records)

Each record is constructed from industry-specific templates to produce
realistic website descriptions, campaign text, and demographic targets.
The generator uses controlled randomness (seeded) so the dataset is
reproducible across runs.
"""

import csv
import os
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Seed for reproducibility — every run produces the same dataset
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PROSPECTS_PATH = DATA_DIR / "prospects.csv"
DEALS_PATH = DATA_DIR / "deal_history.csv"

# ---------------------------------------------------------------------------
# Industry definitions — each entry contains templates for realistic text
# generation tied to that vertical.  The pipeline later uses TF-IDF on
# these text fields, so lexical variety *within* an industry is important
# to avoid degenerate clusters.
# ---------------------------------------------------------------------------
INDUSTRIES: list[dict[str, Any]] = [
    {
        "name": "Technology",
        "companies": [
            "NovaTech Solutions", "CloudPeak Systems", "QuantumByte Inc",
            "DataStream Analytics", "CyberForge Labs", "PixelWave Software",
            "NeuralPath AI", "SkyGrid Computing", "CodeVault Technologies",
            "ByteShift Networks", "Synthetix Data", "Apex Cloud Services",
            "TerraNode Systems", "InfiniByte Corp", "PulseTech Innovations",
            "ZeroDay Security", "MetaFrame Software", "GridIron Analytics",
            "FusionStack Labs", "Quantum Leap Digital", "Archon Systems",
            "VectorPoint AI", "Nimbus Infrastructure", "CipherNet Solutions",
            "Helios Data Group", "Prism Analytics Co", "TitanScale Computing",
            "Orbit Software Labs", "Axiom Cloud Inc", "DeepCore Technologies",
        ],
        "descriptions": [
            "Enterprise SaaS platform providing cloud-native analytics and business intelligence tools for mid-market companies. Serves over 2,000 organizations across North America.",
            "AI-powered cybersecurity firm specializing in threat detection and automated incident response. Their flagship product monitors network traffic in real-time using proprietary machine learning models.",
            "Developer tools company building next-generation CI/CD pipelines and infrastructure-as-code solutions. Focused on reducing deployment times for engineering teams at scale.",
            "Cloud data warehousing platform that unifies structured and unstructured data sources for enterprise analytics. Integrates natively with major BI tools and supports real-time streaming ingestion.",
            "Edge computing startup delivering low-latency inference solutions for IoT devices. Their hardware-software stack powers autonomous vehicle sensors and smart factory applications.",
            "No-code application builder enabling business users to create internal tools and workflows without engineering support. Raised Series B funding and expanding into European markets.",
            "Managed Kubernetes platform simplifying container orchestration for DevOps teams. Offers one-click deployment, auto-scaling, and built-in observability dashboards.",
        ],
        "campaigns": [
            "Launched thought leadership content series on LinkedIn targeting CTOs and VPs of Engineering. Sponsored two major developer conferences with booth and speaking slots.",
            "Running programmatic display campaigns across tech publisher sites with retargeting on product demo page visitors. A/B testing landing pages for free trial conversion.",
            "Executed a webinar series on cloud migration best practices, generating 1,500 qualified leads. Following up with email nurture sequences segmented by company size.",
            "Invested heavily in search engine marketing targeting enterprise software keywords. Complementing with podcast sponsorships on top technology shows.",
            "Launched account-based marketing program targeting Fortune 500 IT departments. Using intent data to personalize outreach across email, display, and direct mail.",
        ],
        "demographics": [
            "IT decision-makers aged 30-50 at mid-to-large enterprises",
            "Software developers and engineering managers at startups and scale-ups",
            "CTOs and CIOs at Fortune 1000 companies evaluating digital transformation",
            "DevOps engineers and platform teams at cloud-native organizations",
            "Technology-forward small business owners seeking automation tools",
        ],
        "revenue_range": (5_000_000, 500_000_000),
        "employee_range": (50, 10_000),
        "ad_spend_range": (100_000, 5_000_000),
        "social_score_range": (50, 95),
    },
    {
        "name": "Entertainment",
        "companies": [
            "Starlight Entertainment", "Vivid Media Group", "Crimson Studios",
            "Thunderbolt Productions", "Horizon Creative Agency",
            "Blue Ember Entertainment", "Velvet Stage Productions",
            "Golden Reel Media", "Echo Chamber Studios", "Neon Pulse Entertainment",
            "Silver Screen Collective", "Waveform Audio Group",
            "Midnight Sun Productions", "Radiant Films", "Mosaic Media Co",
            "Prism Light Studios", "Cascade Entertainment Group",
            "Storyboard Creative", "Limelight Media Inc", "Atlas Productions",
            "Ember & Oak Studios", "Zenith Entertainment Corp",
            "Parallax Media Group", "Kaleidoscope Studios", "Summit Creative Co",
        ],
        "descriptions": [
            "Full-service production company creating original content for streaming platforms and broadcast networks. Recent projects include three Netflix Original series and a Hulu documentary.",
            "Independent music label and artist management firm representing 45 artists across hip-hop, R&B, and pop genres. Operates recording studios in Los Angeles and Atlanta.",
            "Experiential entertainment company producing immersive pop-up events and branded activations for major consumer brands. Known for viral social media moments at their installations.",
            "Gaming studio developing mobile-first casual games with in-app advertising monetization. Their top title has 12 million monthly active users across iOS and Android.",
            "Podcast network producing 30+ shows across true crime, comedy, and culture categories. Monetized through host-read ad placements and premium subscription tiers.",
            "Talent agency representing actors, directors, and writers for film, television, and digital media. Expanding into influencer management and brand partnership brokerage.",
        ],
        "campaigns": [
            "Premiered new series with integrated social media campaign across Instagram, TikTok, and Twitter. Partnered with 50 micro-influencers for authentic audience engagement.",
            "Ran cross-platform media buy including CTV, digital audio, and out-of-home placements for album launch. Generated 10M impressions in first week.",
            "Executed experiential marketing activation at SXSW with branded photo booth and live performances. Captured 5,000 email signups and 2M social impressions.",
            "Launched YouTube pre-roll and mid-roll ad campaign targeting entertainment enthusiasts aged 18-34. Using frequency capping to optimize for view-through conversions.",
            "Deployed programmatic audio ads across Spotify and iHeartRadio targeting fans of similar artists. Retargeting with display ads driving to merchandise store.",
        ],
        "demographics": [
            "Entertainment consumers aged 18-34 with high social media engagement",
            "Music fans and concertgoers aged 21-45 in major metro areas",
            "Streaming platform subscribers interested in original content",
            "Mobile gamers aged 16-35 who engage with in-app advertising",
            "Podcast listeners aged 25-44 interested in culture and lifestyle content",
        ],
        "revenue_range": (2_000_000, 200_000_000),
        "employee_range": (20, 5_000),
        "ad_spend_range": (200_000, 8_000_000),
        "social_score_range": (60, 99),
    },
    {
        "name": "Finance",
        "companies": [
            "Meridian Capital Group", "Ironclad Financial", "Pinnacle Wealth Advisors",
            "Vanguard Fintech Solutions", "Cobalt Banking Systems",
            "Sterling Asset Management", "Citadel Risk Analytics",
            "Granite Financial Corp", "Summit Investment Partners",
            "Osprey Capital Ventures", "Bedrock Insurance Group",
            "Northstar Lending", "Keystone Financial Technologies",
            "Bastion Credit Union", "Paragon Payments Inc",
            "Redwood Wealth Management", "Falcon Financial Services",
            "Evergreen Capital Advisors", "Titanium Trust Corp",
            "Sapphire Banking Solutions", "Cornerstone Credit Partners",
            "Fortify Risk Management", "Bridgewater Analytics Group",
            "Zenith Payments Corp", "Archway Financial Holdings",
        ],
        "descriptions": [
            "Digital-first banking platform offering checking, savings, and investment accounts with zero-fee structure. Serves 800,000 consumers and 50,000 small businesses nationwide.",
            "Wealth management firm providing personalized portfolio strategies for high-net-worth individuals. Manages over $5B in assets with a focus on ESG-aligned investments.",
            "Payment processing company enabling merchants to accept contactless, mobile, and cryptocurrency payments. Processing $2B in annual transaction volume across 15,000 merchant accounts.",
            "Insurance technology company using AI to automate underwriting and claims processing. Partners with 200 insurance carriers to reduce policy issuance time from weeks to minutes.",
            "Commercial lending platform connecting small businesses with capital providers. Uses alternative data scoring models to approve loans for businesses traditional banks decline.",
            "Regulatory technology firm helping financial institutions comply with AML, KYC, and GDPR requirements. Their real-time monitoring platform processes 10M transactions daily.",
        ],
        "campaigns": [
            "Launched brand awareness campaign across financial news sites and business podcasts. Targeting C-suite executives at mid-market companies with content about financial transformation.",
            "Running performance marketing campaigns on Google and LinkedIn targeting small business owners seeking financing. Optimizing for qualified application submissions.",
            "Executed thought leadership strategy with whitepapers and webinars on regulatory compliance trends. Generated 800 MQLs from financial services decision-makers.",
            "Deployed CTV advertising during business news programming targeting affluent investors aged 40-65. Complementing with programmatic display on wealth management content sites.",
            "Launched referral program with existing customers, supported by email marketing and in-app notifications. Referral conversions account for 30% of new customer acquisition.",
        ],
        "demographics": [
            "High-net-worth individuals aged 40-65 with investable assets over $500K",
            "Small business owners aged 30-55 seeking growth capital and financial tools",
            "CFOs and financial controllers at mid-market enterprises",
            "Millennial and Gen-Z consumers seeking digital-first banking experiences",
            "Compliance officers and risk managers at financial institutions",
        ],
        "revenue_range": (10_000_000, 1_000_000_000),
        "employee_range": (100, 20_000),
        "ad_spend_range": (500_000, 10_000_000),
        "social_score_range": (30, 80),
    },
    {
        "name": "Healthcare",
        "companies": [
            "MedVista Health Systems", "CarePoint Technologies",
            "Helix Biomedical", "Pulse Health Analytics",
            "Vitality Therapeutics", "ClearPath Diagnostics",
            "Synapse Health AI", "Everwell Patient Solutions",
            "Genome Insights Corp", "NovaCare Medical Devices",
            "BioHorizon Labs", "Zenith Health Partners",
            "CureLink Telemedicine", "Apex Pharmaceutical Group",
            "Revive Wellness Corp", "MedStream Digital Health",
            "ProHealth Systems Inc", "Catalyst Biotech",
            "Lifeline Health Analytics", "PureForm Medical",
            "WellBridge Health Network", "Guardian Medical Technologies",
            "TrueNorth Diagnostics", "Elara Health Innovations",
            "Summit Care Solutions",
        ],
        "descriptions": [
            "Telehealth platform connecting patients with board-certified physicians for virtual consultations. Processes 50,000 visits per month across 40 states with insurance billing integration.",
            "Medical device manufacturer producing FDA-cleared wearable monitors for chronic disease management. Devices track cardiac rhythm, blood glucose, and respiratory metrics continuously.",
            "Healthcare analytics company providing population health insights to hospital systems and payers. Their predictive models identify at-risk patients 90 days before acute events.",
            "Pharmaceutical company developing novel biologics for autoimmune disorders. Two candidates in Phase III clinical trials with FDA fast-track designation.",
            "Digital health startup offering AI-powered symptom triage and care navigation for employer-sponsored health plans. Reducing unnecessary ER visits by 35% for enrolled populations.",
            "Clinical trial management platform streamlining patient recruitment, data collection, and regulatory submissions. Used by 150 research sites across oncology and rare disease studies.",
        ],
        "campaigns": [
            "Launched HCP-targeted digital campaign across medical journals and physician social networks. Using NPI-level targeting to reach specialists in cardiology and endocrinology.",
            "Running disease awareness campaign for patients with autoimmune conditions across health content sites and support communities. Driving traffic to educational resources and clinical trial finder.",
            "Executed multi-channel campaign targeting hospital administrators at HIMSS conference. Combined booth presence with pre-show email outreach and post-show retargeting.",
            "Deployed programmatic campaigns across health and wellness sites targeting HR benefits decision-makers. Promoting employer health plan integration with ROI case studies.",
            "Launched physician education webinar series on remote patient monitoring best practices. CME-accredited content generating leads from primary care and specialist practices.",
        ],
        "demographics": [
            "Healthcare providers and physicians aged 35-60 in hospital and clinic settings",
            "Hospital administrators and health system C-suite executives",
            "Patients aged 40-70 managing chronic conditions seeking digital health tools",
            "HR and benefits managers at companies with 500+ employees",
            "Clinical researchers and principal investigators at academic medical centers",
        ],
        "revenue_range": (5_000_000, 800_000_000),
        "employee_range": (30, 15_000),
        "ad_spend_range": (200_000, 6_000_000),
        "social_score_range": (20, 70),
    },
    {
        "name": "Retail & E-commerce",
        "companies": [
            "UrbanThread Apparel", "FreshCart Groceries", "LuxeHome Interiors",
            "PeakGear Outdoors", "GlowUp Beauty Co", "SwiftShip Logistics",
            "CraftedGoods Marketplace", "TrendSetters Fashion",
            "PantryPlus Delivery", "EcoBloom Sustainable Goods",
            "MetroStyle Boutique", "GadgetBox Electronics",
            "VintageVault Collectibles", "PureLeaf Organics",
            "SnapDeal Commerce", "WildTrail Adventure Gear",
            "BrightNest Home Goods", "FitForm Activewear",
            "ArtisanAlley Crafts", "SparkJoy Gifts",
            "HarvestTable Foods", "UrbanRoots Garden Supply",
            "CloudCloset Fashion Tech", "NimbleMart Retail",
            "Bloom & Barrel Co",
        ],
        "descriptions": [
            "Direct-to-consumer fashion brand specializing in sustainable and ethically sourced apparel. Operates online and through 12 retail locations in major US cities with a loyalty program of 200,000 members.",
            "Online grocery delivery platform serving metropolitan areas with same-day and next-day delivery. Partners with 500 local farms and producers for farm-to-door freshness.",
            "Luxury home furnishings retailer with an e-commerce platform and showroom model. Uses AR technology to let customers visualize furniture in their spaces before purchasing.",
            "Outdoor recreation retailer selling camping, hiking, and climbing gear through a curated online marketplace. Known for expert product reviews and a community-driven recommendation engine.",
            "Clean beauty brand selling skincare and cosmetics made from natural ingredients. DTC model with a subscription box generating 40% of recurring revenue.",
            "Multi-brand e-commerce aggregator offering electronics, gadgets, and accessories at competitive prices. Uses dynamic pricing algorithms and flash sale events to drive conversion.",
        ],
        "campaigns": [
            "Running Instagram and TikTok shopping campaigns with shoppable posts and influencer collaborations. Targeting fashion-forward consumers aged 18-35 with seasonal collection launches.",
            "Launched Google Shopping and Performance Max campaigns optimized for ROAS. Retargeting cart abandoners with dynamic product ads across Meta and programmatic display.",
            "Executed holiday season media blitz across CTV, digital audio, and social media. Increased ad spend 40% during Black Friday to Cyber Monday window with record conversion rates.",
            "Deployed email and SMS marketing automation with personalized product recommendations based on browsing history. Achieving 25% open rates and 4% click-through rates.",
            "Launched loyalty program marketing campaign rewarding repeat customers with points, early access, and exclusive discounts. Supported by in-app push notifications and email.",
        ],
        "demographics": [
            "Millennial and Gen-Z consumers aged 18-35 with high online shopping frequency",
            "Affluent homeowners aged 30-55 interested in home decor and furnishings",
            "Health-conscious consumers aged 25-45 seeking organic and sustainable products",
            "Outdoor enthusiasts aged 22-50 with active lifestyles",
            "Value-driven shoppers aged 25-50 comparing prices across platforms",
        ],
        "revenue_range": (1_000_000, 300_000_000),
        "employee_range": (10, 8_000),
        "ad_spend_range": (50_000, 4_000_000),
        "social_score_range": (55, 98),
    },
    {
        "name": "Automotive",
        "companies": [
            "VoltDrive Motors", "ApexAuto Group", "TerraMotion EV",
            "IronHorse Fleet Services", "ClearRoad Technologies",
            "Velocity Auto Parts", "GreenWheel Electric",
            "PrecisionDrive Systems", "RoadRunner Logistics",
            "TitanTruck Commercial", "AutoPilot Innovations",
            "ElectraFleet Solutions", "CruiseControl Software",
            "SteelPath Auto Group", "MotionTech Mobility",
        ],
        "descriptions": [
            "Electric vehicle manufacturer producing affordable EVs for the mass market. Their compact sedan starts at $28,000 and offers 300 miles of range with a nationwide charging network.",
            "Automotive dealership group operating 35 locations across the Southeast. Sells new and certified pre-owned vehicles with an integrated online purchasing experience.",
            "Fleet management software company providing GPS tracking, maintenance scheduling, and fuel optimization for commercial vehicle operators managing 100+ vehicle fleets.",
            "Autonomous driving technology company developing Level 4 self-driving systems for last-mile delivery vehicles. Currently operating pilot programs in three metro areas.",
            "Aftermarket auto parts e-commerce platform with 2 million SKUs and next-day delivery. Serving both professional mechanics and DIY enthusiasts nationwide.",
            "Connected car platform providing in-vehicle infotainment, OTA updates, and predictive maintenance alerts. Integrated into four major OEM vehicle lineups.",
        ],
        "campaigns": [
            "Launched multi-channel brand campaign across CTV, digital video, and out-of-home highlighting EV range and affordability. Targeting eco-conscious consumers aged 25-50 in urban markets.",
            "Running local search and display campaigns targeting in-market car buyers within 50-mile radius of dealership locations. Using auto intender data for audience targeting.",
            "Executed B2B campaign targeting fleet managers through LinkedIn, industry publications, and trade show sponsorships. Promoting TCO calculator and ROI case studies.",
            "Deployed dynamic retargeting campaigns for website visitors who configured vehicles but did not schedule test drives. Offering limited-time incentives to drive showroom traffic.",
            "Launched co-branded partnership campaign with charging network provider. Cross-promoting EV ownership benefits through email, social media, and in-dealership signage.",
        ],
        "demographics": [
            "In-market car buyers aged 25-55 researching vehicles online",
            "Fleet managers and logistics directors at companies with 50+ vehicles",
            "Eco-conscious consumers aged 22-45 considering electric vehicles",
            "Automotive enthusiasts and DIY mechanics aged 20-50",
            "Commercial transportation operators and owner-operators",
        ],
        "revenue_range": (10_000_000, 2_000_000_000),
        "employee_range": (50, 25_000),
        "ad_spend_range": (300_000, 15_000_000),
        "social_score_range": (25, 75),
    },
    {
        "name": "Food & Beverage",
        "companies": [
            "BrightBrew Coffee", "HarvestMoon Foods", "PureSip Beverages",
            "TerraGrain Baking Co", "FreshFusion Juicery",
            "GoldenHarvest Snacks", "CraftBrew Collective",
            "NourishPlate Meals", "SunRipe Produce Co",
            "MapleLine Confections", "OceanCatch Seafood",
            "VineRoots Winery", "SpiceLane Foods",
            "MeadowFarm Dairy", "UrbanBite Restaurants",
        ],
        "descriptions": [
            "Specialty coffee roaster and cafe chain with 45 locations and a thriving DTC subscription business. Sources beans directly from farms in Colombia, Ethiopia, and Guatemala.",
            "Plant-based food company producing meat alternatives sold in 8,000 grocery stores nationwide. Recently launched a line of frozen meals targeting health-conscious families.",
            "Craft brewery operating a taproom, regional distribution across 12 states, and a beer-of-the-month subscription club. Known for experimental IPAs and seasonal limited releases.",
            "Premium snack brand offering better-for-you chips, crackers, and trail mixes made with clean ingredients. Sold through natural grocery chains, Amazon, and their own e-commerce site.",
            "Farm-to-table restaurant group operating six locations with a focus on locally sourced, seasonal menus. Expanding into catering and meal kit delivery for the home cook market.",
            "Functional beverage company producing adaptogen-infused drinks targeting wellness enthusiasts. Available in specialty retail and through a DTC subscription model.",
        ],
        "campaigns": [
            "Running influencer marketing campaign on Instagram and TikTok with food bloggers and wellness creators. Sampling program at farmers markets and fitness events in target metro areas.",
            "Launched in-store sampling and retail media campaigns at Whole Foods and Sprouts locations. Complementing with digital coupons and loyalty program integrations.",
            "Executed seasonal campaign around summer grilling season across food media sites, YouTube pre-roll, and connected TV. Partnering with celebrity chef for recipe content series.",
            "Deployed geo-targeted mobile advertising within 3-mile radius of retail locations driving foot traffic. Using location-based attribution to measure in-store visit lift.",
            "Launched sustainability-focused brand campaign highlighting ethical sourcing and carbon-neutral operations. Running across premium digital publishers targeting conscious consumers.",
        ],
        "demographics": [
            "Health-conscious consumers aged 25-45 seeking clean-label food products",
            "Foodies and home cooks aged 28-50 interested in premium ingredients",
            "Craft beverage enthusiasts aged 21-40 in urban and suburban markets",
            "Families aged 30-50 seeking convenient and nutritious meal solutions",
            "Wellness-focused millennials and Gen-Z consumers aged 18-35",
        ],
        "revenue_range": (500_000, 150_000_000),
        "employee_range": (10, 3_000),
        "ad_spend_range": (25_000, 2_000_000),
        "social_score_range": (45, 95),
    },
    {
        "name": "Real Estate",
        "companies": [
            "Skyline Properties", "Cornerstone Realty Group",
            "UrbanNest Developments", "ClearTitle Mortgage",
            "PrimeLocation Brokerage", "GreenBuild Construction",
            "MetroLiving Apartments", "FairHaven Real Estate",
            "BlueDoor Home Loans", "Landmark Commercial Realty",
            "SilverKey Property Management", "HorizonView Developments",
            "TrueHome Inspections", "ParkPlace Luxury Realty",
            "CedarPoint Properties",
        ],
        "descriptions": [
            "Residential real estate brokerage with 200 agents across three metro markets. Differentiates with proprietary market analytics tools and a high-touch client experience.",
            "Commercial real estate development firm specializing in mixed-use urban projects. Currently developing $500M in projects across the Sun Belt with a focus on sustainable design.",
            "Property technology company providing an all-in-one platform for property managers to handle leasing, maintenance, and tenant communications. Managing 50,000 rental units.",
            "Mortgage lending company offering conventional, FHA, and VA loans with a fully digital application process. Average closing time of 21 days, 40% below industry average.",
            "Luxury real estate firm representing sellers and buyers of properties valued at $2M and above. Operates in New York, Miami, Los Angeles, and Aspen markets.",
            "Build-to-rent housing developer constructing single-family rental communities in high-growth suburban markets. Portfolio of 3,000 homes across Texas, Florida, and Georgia.",
        ],
        "campaigns": [
            "Running hyperlocal digital campaigns targeting homebuyers in specific zip codes with Google Search and display. Using MLS data to create dynamic ads featuring available listings.",
            "Launched luxury brand campaign across premium digital publishers, private aviation magazines, and wealth management event sponsorships. Targeting ultra-high-net-worth individuals.",
            "Executed landlord and property manager acquisition campaign on LinkedIn and industry publications. Offering free ROI analysis and platform demo for portfolios of 50+ units.",
            "Deployed first-time homebuyer education campaign with video content, webinars, and downloadable guides. Running across social media and search targeting millennials entering the market.",
            "Launched neighborhood-level content marketing strategy with hyper-local blog posts and video tours. Optimized for long-tail real estate search queries in target markets.",
        ],
        "demographics": [
            "First-time homebuyers aged 28-40 in major metropolitan areas",
            "Real estate investors and landlords managing rental property portfolios",
            "Ultra-high-net-worth individuals seeking luxury properties",
            "Property managers at firms overseeing 100+ rental units",
            "Commercial tenants and businesses seeking office and retail space",
        ],
        "revenue_range": (2_000_000, 500_000_000),
        "employee_range": (15, 5_000),
        "ad_spend_range": (100_000, 3_000_000),
        "social_score_range": (30, 80),
    },
    {
        "name": "Education",
        "companies": [
            "BrightPath Learning", "SkillForge Academy",
            "MindSpark EdTech", "EduVista Online",
            "NextGen Tutoring", "Campus Connect Solutions",
            "LearnLoop Technologies", "AcademicEdge Prep",
            "CodeCraft Bootcamp", "ScholarsGate Publishing",
            "CuriousMind Labs", "TutorBridge Platform",
            "Elevate Learning Group", "PenPoint EdMedia",
            "FutureReady Schools",
        ],
        "descriptions": [
            "Online learning platform offering professional development courses in data science, AI, and cloud computing. Serves 500,000 learners with content from industry practitioners and university faculty.",
            "K-12 education technology company providing adaptive learning software for math and reading. Used by 5,000 school districts to personalize instruction for diverse learners.",
            "Coding bootcamp operating in-person and online programs in full-stack development, data engineering, and UX design. 90% job placement rate within six months of graduation.",
            "Higher education enrollment management platform helping universities optimize recruitment, admissions, and student retention. Integrating predictive analytics and CRM automation.",
            "Corporate training company delivering leadership development, compliance, and skills training for Fortune 500 companies. Blended learning model combining virtual workshops and e-learning.",
            "Educational content publisher producing textbooks, digital courseware, and assessment tools for the college market. Transitioning to a subscription-based digital-first delivery model.",
        ],
        "campaigns": [
            "Running prospecting campaigns on LinkedIn and Google targeting working professionals seeking career advancement through online education. Promoting free introductory courses as lead magnets.",
            "Launched back-to-school campaign across social media and display targeting parents of K-12 students. Highlighting adaptive learning outcomes and teacher testimonials.",
            "Executed university enrollment marketing campaign across search, social, and CTV targeting high school juniors and seniors. Using student ambassador content for authenticity.",
            "Deployed B2B content marketing strategy targeting L&D directors and CHROs. Publishing ROI studies and hosting roundtable events on workforce upskilling trends.",
            "Launched scholarship awareness campaign targeting underrepresented communities through community organizations, social media, and local radio. Driving applications for funded programs.",
        ],
        "demographics": [
            "Working professionals aged 25-45 seeking career advancement through education",
            "Parents of K-12 students aged 30-50 investing in children's education",
            "High school students aged 16-18 and their families planning for college",
            "L&D leaders and HR executives at enterprises with 1000+ employees",
            "Career changers aged 25-40 exploring bootcamp and certification programs",
        ],
        "revenue_range": (1_000_000, 200_000_000),
        "employee_range": (20, 3_000),
        "ad_spend_range": (50_000, 2_500_000),
        "social_score_range": (35, 85),
    },
    {
        "name": "Travel & Hospitality",
        "companies": [
            "Wanderlust Resorts", "SkyBound Airlines",
            "TrailHead Adventures", "LuxeStay Hotels",
            "CoastalBreeze Cruises", "NomadPack Travel Tech",
            "SerenitySpas Wellness", "JetStream Tours",
            "AlpineRidge Lodges", "UrbanExplorer Hostels",
            "Voyager Booking Platform", "SummitView Retreats",
            "GoldenSands Resorts", "CulturalBridge Tours",
            "HarborLight Hospitality",
        ],
        "descriptions": [
            "Boutique hotel chain operating 20 properties in coastal and mountain destinations. Each property features locally inspired design, farm-to-table dining, and curated experiential programming.",
            "Online travel agency specializing in adventure and experiential travel packages. Offers guided tours, activity bookings, and trip planning tools for 50 destinations worldwide.",
            "Corporate travel management platform providing booking, expense tracking, and policy compliance tools for business travelers. Managing travel programs for 800 companies.",
            "Luxury cruise line operating expedition voyages to Antarctica, the Arctic, and remote Pacific islands. Ships carry 200 guests with all-inclusive pricing and expert naturalist guides.",
            "Wellness retreat company operating immersive programs in Bali, Costa Rica, and Sedona. Combining yoga, meditation, and nutrition with luxury accommodations.",
            "Vacation rental management company operating 2,000 short-term rental properties across popular tourist destinations. Providing full-service management including dynamic pricing and guest communication.",
        ],
        "campaigns": [
            "Launched aspirational brand campaign across Instagram, Pinterest, and CTV featuring cinematic destination content. Targeting affluent travelers aged 30-55 dreaming of their next trip.",
            "Running search and metasearch campaigns across Google, Kayak, and TripAdvisor targeting price-sensitive travelers comparing options. Optimizing bids by destination and travel dates.",
            "Executed email re-engagement campaign targeting past guests with personalized offers based on previous stays. Achieving 35% open rates with loyalty program incentives.",
            "Deployed geo-targeted mobile campaigns at airports and transit hubs promoting last-minute deals and upgrades. Using proximity data to reach travelers in real-time.",
            "Launched corporate travel program campaign on LinkedIn targeting travel managers and procurement teams. Highlighting cost savings, duty-of-care features, and traveler satisfaction scores.",
        ],
        "demographics": [
            "Affluent leisure travelers aged 30-60 with annual travel budgets over $10K",
            "Business travelers aged 28-55 managing corporate travel programs",
            "Adventure seekers aged 22-40 interested in experiential and active travel",
            "Wellness-focused travelers aged 30-50 seeking retreat experiences",
            "Families with children aged 5-17 planning vacation getaways",
        ],
        "revenue_range": (2_000_000, 400_000_000),
        "employee_range": (25, 8_000),
        "ad_spend_range": (100_000, 5_000_000),
        "social_score_range": (40, 92),
    },
]


def _rand_in_range(low: int, high: int) -> int:
    """Return a random integer between low and high, inclusive."""
    return random.randint(low, high)


def _generate_prospects(n: int = 500) -> list[dict[str, Any]]:
    """
    Build *n* synthetic prospect records by sampling from industry templates.

    Each record receives a unique company name, a randomly selected
    description/campaign/demographic string from its industry pool, and
    numeric features drawn uniformly from the industry's configured ranges.
    """
    prospects: list[dict[str, Any]] = []

    # ---- distribute n records across industries roughly evenly ----------
    per_industry = n // len(INDUSTRIES)
    remainder = n % len(INDUSTRIES)

    for idx, industry in enumerate(INDUSTRIES):
        # The first `remainder` industries each get one extra record
        count = per_industry + (1 if idx < remainder else 0)
        companies = list(industry["companies"])  # copy so we can pop
        random.shuffle(companies)

        for i in range(count):
            # Cycle through company names if we need more records than names
            company = companies[i % len(companies)]
            # Append a suffix if we've cycled through all company names to
            # ensure uniqueness across the full dataset
            if i >= len(companies):
                company = f"{company} ({i // len(companies) + 1})"

            rev_lo, rev_hi = industry["revenue_range"]
            emp_lo, emp_hi = industry["employee_range"]
            ads_lo, ads_hi = industry["ad_spend_range"]
            soc_lo, soc_hi = industry["social_score_range"]

            prospects.append(
                {
                    "company_name": company,
                    "industry": industry["name"],
                    "annual_revenue": _rand_in_range(rev_lo, rev_hi),
                    "employee_count": _rand_in_range(emp_lo, emp_hi),
                    "website_description": random.choice(
                        industry["descriptions"]
                    ),
                    "recent_campaigns": random.choice(industry["campaigns"]),
                    "social_presence_score": _rand_in_range(soc_lo, soc_hi),
                    "current_ad_spend_estimate": _rand_in_range(ads_lo, ads_hi),
                    "target_demographics": random.choice(
                        industry["demographics"]
                    ),
                }
            )

    random.shuffle(prospects)
    return prospects


def _generate_deals(
    prospects: list[dict[str, Any]], n: int = 200
) -> list[dict[str, Any]]:
    """
    Build *n* synthetic deal-history records by sampling from the prospect
    list.  Deal values, channels, outcomes, and cycle lengths are randomised
    but correlated with the prospect's numeric profile (bigger companies
    tend to have larger deals and longer cycles).
    """
    # Channel options that a media sales team would recognise
    channel_options = [
        "Programmatic Display", "CTV/OTT", "Paid Social", "Podcast Audio",
        "Search/SEM", "Native Content", "Out-of-Home", "Email Marketing",
        "Influencer", "Direct Mail", "Digital Video", "Print",
        "Retail Media", "Branded Content", "Events & Sponsorships",
    ]

    deals: list[dict[str, Any]] = []
    sampled = random.choices(prospects, k=n)

    for prospect in sampled:
        # Deal value is loosely proportional to the prospect's ad spend,
        # adding noise so the model has something interesting to learn
        base = prospect["current_ad_spend_estimate"]
        deal_value = int(base * random.uniform(0.1, 0.6))

        # Select 1-4 channels for this deal
        num_channels = random.randint(1, 4)
        channels = ", ".join(random.sample(channel_options, num_channels))

        # Win rate influenced by social score (higher engagement → higher
        # close rate).  This creates a learnable signal.
        win_prob = 0.3 + (prospect["social_presence_score"] / 100) * 0.4
        outcome = "won" if random.random() < win_prob else "lost"

        # Larger companies → longer sales cycles
        base_cycle = 30 + (prospect["employee_count"] / 1_000) * 20
        sales_cycle = int(base_cycle * random.uniform(0.5, 2.0))

        deals.append(
            {
                "company_name": prospect["company_name"],
                "deal_value": deal_value,
                "channels_used": channels,
                "outcome": outcome,
                "sales_cycle_days": min(sales_cycle, 365),
            }
        )

    return deals


def main() -> None:
    """Generate both CSV files and write them to the data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- prospects.csv --------------------------------------------------
    prospects = _generate_prospects(500)
    fieldnames_p = [
        "company_name", "industry", "annual_revenue", "employee_count",
        "website_description", "recent_campaigns", "social_presence_score",
        "current_ad_spend_estimate", "target_demographics",
    ]
    with open(PROSPECTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_p)
        writer.writeheader()
        writer.writerows(prospects)
    print(f"Wrote {len(prospects)} prospects to {PROSPECTS_PATH}")

    # ---- deal_history.csv -----------------------------------------------
    deals = _generate_deals(prospects, 200)
    fieldnames_d = [
        "company_name", "deal_value", "channels_used", "outcome",
        "sales_cycle_days",
    ]
    with open(DEALS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_d)
        writer.writeheader()
        writer.writerows(deals)
    print(f"Wrote {len(deals)} deals to {DEALS_PATH}")


if __name__ == "__main__":
    main()
