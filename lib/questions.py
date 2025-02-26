import re


class QuestionExtractor:
    def __init__(self):
        self.patterns = [
            {
                "name": "indicator_compare",
                "pattern": re.compile(
                    r'Which of the companies had the (?P<comparison>highest|lowest) (?P<metric>.+?) in (?P<currency>\S+) at the end of the period listed in annual report: (?P<companies>.+?)\?'
                ),
            },
            {
                "name": "fin_metric",
                "pattern": re.compile(
                    r'(?:What was the|According to the annual report, what is the) (?P<metric>.+?) for (?P<company>.+?) (?:according to the annual report|\(within the last period or at the end of the last period\))[^?]*\?'
                ),
            },
            {
                "name": "latest_merger_entity",
                "pattern": re.compile(
                    r'What was the latest merger or acquisition that (?P<company>.+?) was involved in\?'
                ),
                "metric": "latest merger or acquisition"
            },
            {
                "name": "merger_mention",
                "pattern": re.compile(
                    r'Did (?P<company>.+?) mention any mergers or acquisitions in the annual report\?'
                ),
                "metric": "merger mention"
            },
            {
                "name": "compensation",
                "pattern": re.compile(
                    r'What was the largest single spending of (?P<company>.+?) on executive compensation in (?P<currency>\S+)\?'
                ),
                "metric": "largest single spending on executive compensation"
            },
            {
                "name": "leadership_removed",
                "pattern": re.compile(
                    r'What are the names of all executives removed from their positions in (?P<company>.+?)\?'
                ),
                "metric": "names of all executives removed"
            },
            {
                "name": "leadership_added",
                "pattern": re.compile(
                    r'What are the names of all new executives that took on new leadership positions in (?P<company>.+?)\?'
                ),
                "metric": "names of all new executives that took on new leadership positions"
            },
            {
                "name": "leadership_changed",
                "pattern": re.compile(
                    r'Which leadership \*\*positions\*\* changed at (?P<company>.+?) in the reporting period\?'
                ),
                "metric": "leadership positions changed"
            },
            {
                "name": "leadership_announcement",
                "pattern": re.compile(
                    r'Did (?P<company>.+?) announce any changes to its executive team in the annual report\?'
                ),
                "metric": "changes to its executive team"
            },
            {
                "name": "layoffs",
                "pattern": re.compile(
                    r'(?:How many employees were laid off by|What is the total number of employees let go by) (?P<company>.+?) (?:during the period covered by the annual report|according to the annual report)\?'
                ),
                "metric": "total number of layoffs"
            },
            {
                "name": "product_names",
                "pattern": re.compile(
                    r'What are the names of new products launched by (?P<company>.+?) as mentioned in the annual report\?'
                ),
                "metric": "new products launched names"
            },
            {
                "name": "last_product",
                "pattern": re.compile(
                    r'What is the name of the last product launched by (?P<company>.+?) as mentioned in the annual report\?'
                ),
                "metric": "last product launch"
            },
            {
                "name": "product_announcement",
                "pattern": re.compile(
                    r'Did (?P<company>.+?) announce any new product launches in the annual report\?'
                ),
                "metric": "new product launch announcement"
            },
            {
                "name": "has_regulatory_or_litigation_issues",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) mention any ongoing litigation or regulatory inquiries\?"
                ),
                "metric": "ongoing litigation or regulatory inquiries"
            },
            {
                "name": "has_capital_structure_changes",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) report any changes to its capital structure\?"
                ),
                "metric": "changes to its capital structure"
            },
            {
                "name": "has_share_buyback_plans",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) announce a share buyback plan in the annual report\?"
                ),
                "metric": "share buyback plan"
            },
            {
                "name": "has_dividend_policy_changes",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) announce any changes to its dividend policy in the annual report\?"
                ),
                "metric": "changes to its dividend policy"
            },
            {
                "name": "has_strategic_restructuring",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) detail any restructuring plans in the latest filing\?"
                ),
                "metric": "restructuring plans"
            },
            {
                "name": "has_supply_chain_disruptions",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) report any supply chain disruptions in the annual report\?"
                ),
                "metric": "supply chain disruptions"
            },
            {
                "name": "has_esg_initiatives",
                "pattern": re.compile(
                    r"Did (?P<company>.+?) outline any new ESG initiatives in the annual report\?"
                ),
                "metric": "new ESG initiatives"
            },
            {
                "name": "industry_metric",
                "pattern": re.compile(
                    r'(?:What was the value of (?P<metric>.+?) of (?P<company>.+?) at the end of the period listed in annual report\?|For (?P<company_alt>.+?), what was the value of (?P<metric_alt>.+?) at the end of the period listed in annual report\?)'
                ),
            }
        ]

    def extract(self, question: str) -> dict:
        """
        Given a question string, attempts to extract:
          - question_type (as defined by our generators)
          - metric (the key performance indicator or topic in question)
          - companies (list of company names mentioned)
          - currency (if applicable)
          - comparison (if the question asks for highest/lowest)
          - category (which generator pattern matched)
        Returns a dictionary with all extracted fields.
        """
        result = {
            "original_question": question,
            "metric": None,
            "companies": [],
            "currency": None,
            "comparison": None,
            "category": None
        }
        for entry in self.patterns:
            match = entry["pattern"].search(question)
            if match:
                if "metric" in entry:
                    result["metric"] = entry["metric"]
                # Extract any named groups from the match.
                groups = match.groupdict()
                # Extract company names.
                companies = []
                if groups.get("company"):
                    companies.append(groups["company"].strip())
                if groups.get("companies"):
                    # Companies might be a comma-separated string (sometimes quoted).
                    comp_list = [c.strip().strip('"') for c in groups["companies"].split(",")]
                    companies.extend(comp_list)
                if groups.get("company_alt"):
                    companies.append(groups["company_alt"].strip())
                result["companies"] = companies

                # If the metric is directly provided in the question.
                if groups.get("metric"):
                    result["metric"] = groups["metric"].strip()
                if groups.get("metric_alt"):
                    result["metric"] = groups["metric_alt"].strip()

                # Extract currency if present.
                if groups.get("currency"):
                    result["currency"] = groups["currency"].strip()

                # Extract comparison if present.
                if groups.get("comparison"):
                    result["comparison"] = groups["comparison"].strip()

                # Mark the category (which pattern matched).
                result["category"] = entry["name"]
                break  # Stop after the first matching pattern.
        return result

    def all_metrix(self) -> list[str]:
        return [
            # Indicator compare
            "total revenue",
            "net income",
            "total assets",
            # Financial KPIs
            "Operating income",
            "Gross margin (%)",
            "Operating margin (%)",
            "EPS (earnings per share)",
            "EBITDA",
            "Capital expenditures",
            "Cash flow from operations",
            "Long-term debt",
            "Shareholders' equity",
            "Dividend per share",

            # Merger questions
            "latest merger or acquisition",
            "merger mention",

            # Executive compensation
            "largest single spending on executive compensation",

            # Leadership related
            "names of all executives removed",
            "names of all new executives that took on new leadership positions",
            "leadership positions changed",
            "changes to its executive team",

            # Layoffs
            "total number of layoffs",

            # Product launches
            "new products launched names",
            "last product launch",
            "new product launch announcement",

            # Metadata Boolean Questions
            "ongoing litigation or regulatory inquiries",
            "changes to its capital structure",
            "share buyback plan",
            "changes to its dividend policy",
            "restructuring plans",
            "supply chain disruptions",
            "new ESG initiatives",
        ]

    @staticmethod
    def get_synonyms(metric: str) -> list:
        """
        Given a metric string, returns a list of 5-10 synonym phrases
        with the same meaning.
        """
        synonyms_mapping = {
            "total revenue": [
                "gross revenue",
                "sales revenue",
                "overall revenue",
                "total sales",
                "revenue total",
                "income from operations"
            ],
            "net income": [
                "net profit",
                "bottom line",
                "net earnings",
                "after-tax profit",
                "earnings after tax"
            ],
            "total assets": [
                "aggregate assets",
                "overall assets",
                "asset total",
                "total resources"
            ],
            "operating income": [
                "operating profit",
                "core earnings",
                "operational income",
                "EBIT"
            ],
            "gross margin (%)": [
                "gross profit margin",
                "gross margin percentage",
                "percentage gross profit",
                "profit margin (gross)"
            ],
            "operating margin (%)": [
                "operating profit margin",
                "operational margin",
                "profit margin (operating)",
                "percentage operating profit"
            ],
            "eps (earnings per share)": [
                "earnings per share",
                "EPS",
                "per share earnings",
                "net income per share"
            ],
            "ebitda": [
                "earnings before interest, taxes, depreciation, and amortization",
                "EBITDA margin",
                "adjusted EBITDA"
            ],
            "capital expenditures": [
                "capex",
                "capital spending",
                "investment in fixed assets",
                "capital outlay"
            ],
            "cash flow from operations": [
                "operating cash flow",
                "cash flow from operating activities",
                "operational cash flow"
            ],
            "long-term debt": [
                "non-current liabilities",
                "long-run debt",
                "debt maturing after one year"
            ],
            "shareholders' equity": [
                "stockholders' equity",
                "owners' equity",
                "equity capital",
                "net assets"
            ],
            "dividend per share": [
                "dividend payout per share",
                "per share dividend",
                "dividend yield per share"
            ],
            "latest merger or acquisition": [
                "most recent merger or acquisition",
                "latest M&A",
                "recent merger/acquisition",
                "newest merger or acquisition"
            ],
            "merger mention": [
                "merger reference",
                "acquisition mention",
                "merger/acquisition mention",
                "M&A mention"
            ],
            "largest single spending on executive compensation": [
                "highest executive compensation expense",
                "maximal executive pay expenditure",
                "largest executive remuneration",
                "largest executive pay outlay"
            ],
            "names of all executives removed": [
                "executives ousted",
                "executives terminated",
                "list of removed executives",
                "dismissed executives"
            ],
            "names of all new executives that took on new leadership positions": [
                "new executive appointments",
                "newly appointed executives",
                "list of new leaders",
                "new leadership hires"
            ],
            "leadership positions changed": [
                "changed leadership roles",
                "modified executive positions",
                "altered leadership roles",
                "leadership role modifications"
            ],
            "changes to its executive team": [
                "executive team changes",
                "leadership changes",
                "restructuring of the executive team",
                "alterations in the executive team"
            ],
            "total number of layoffs": [
                "layoff count",
                "number of employees laid off",
                "job cuts",
                "total layoffs"
            ],
            "new products launched names": [
                "names of new products",
                "launched product names",
                "new product titles",
                "product launch names"
            ],
            "last product launch": [
                "most recent product launch",
                "latest product launch",
                "final product launch",
                "newest product release"
            ],
            "new product launch announcement": [
                "announcement of new product launch",
                "new product launch declaration",
                "product launch announcement",
                "new product introduction"
            ],
            "ongoing litigation or regulatory inquiries": [
                "current litigation or regulatory issues",
                "active legal or regulatory proceedings",
                "ongoing legal challenges",
                "current regulatory inquiries"
            ],
            "changes to its capital structure": [
                "capital structure modifications",
                "alterations to capital composition",
                "capital structure adjustments",
                "changes in financing structure"
            ],
            "share buyback plan": [
                "stock repurchase plan",
                "share repurchase program",
                "buyback strategy",
                "share repurchase scheme"
            ],
            "changes to its dividend policy": [
                "dividend policy modifications",
                "alterations in dividend distribution",
                "changes in dividend payouts",
                "dividend strategy changes"
            ],
            "restructuring plans": [
                "restructuring initiatives",
                "restructuring strategy",
                "organizational restructuring",
                "restructuring proposals"
            ],
            "supply chain disruptions": [
                "supply chain interruptions",
                "breakdowns in the supply chain",
                "logistical disruptions",
                "supply chain disturbances"
            ],
            "new esg initiatives": [
                "new environmental, social, and governance initiatives",
                "new esg programs",
                "sustainability initiatives",
                "new sustainability programs"
            ]
        }

        # Standardize the input metric string (lowercase, strip spaces)
        key = metric.lower().strip()
        # Look for a matching key in our synonyms mapping (using substring check)
        for k, syns in synonyms_mapping.items():
            if k in key:
                return syns
        return []  # Return empty list if no match is found

    @staticmethod
    def industry_metrics(industry) -> list[str]:
        industry_metrics = {
            "Technology": [
                "Number of patents at year-end",
                "Total capitalized R&D expenditure",
                "Total expensed R&D expenditure",
                "End-of-year tech staff headcount",
                "End-of-year total headcount",
                "Annual recurring revenue (ARR)",
                "Total intangible assets (IP valuation)",
                "Number of active software licenses",
                "Data center capacity (MW)",
                "Data center capacity (sq. ft.)",
                "Cloud storage capacity (TB)",
                "End-of-period market capitalization",
                "Year-end customer base",
                "Year-end user base"
            ],
            "Financial Services": [
                "Total assets on balance sheet at year-end",
                "Total deposits at year-end",
                "Loans outstanding at year-end",
                "Assets under management (AUM)",
                "Non-performing loan ratio (NPL) at year-end",
                "Tier 1 capital ratio at year-end",
                "Number of customer accounts at year-end",
                "Branch count at year-end",
                "End-of-year net interest margin (NIM)",
                "Return on equity (ROE) at year-end"
            ],
            "Healthcare": [
                "Number of hospital beds at year-end",
                "Number of owned clinics at year-end",
                "Number of managed clinics at year-end",
                "Active patient count (registered patients)",
                "Value of medical equipment (balance sheet)",
                "End-of-year bed occupancy rate",
                "Number of healthcare professionals on staff",
                "Number of laboratories at year-end",
                "Number of diagnostic centers at year-end",
                "Healthcare plan memberships (if applicable)",
                "Outstanding insurance claims (if applicable)",
                "R&D pipeline (number of therapies in phases)"
            ],
            "Automotive": [
                "Vehicle production capacity (units/year)",
                "Inventory of finished vehicles at year-end",
                "Global dealership network size",
                "Number of electric models available",
                "Number of hybrid models available",
                "Battery production capacity (if applicable)",
                "End-of-year automotive patent portfolio",
                "End-of-period market share (by units sold)",
                "Number of EV charging stations in network",
                "Year-end fleet average CO₂ emissions",
                "R&D workforce headcount"
            ],
            "Retail": [
                "Number of stores at year-end",
                "Total store floor area (sqm)",
                "Total store floor area (sq. ft.)",
                "Value of inventory on hand at year-end",
                "Number of distribution centers at year-end",
                "Number of fulfillment centers at year-end",
                "Loyalty program membership at year-end",
                "Online active customer accounts",
                "E-commerce active customer accounts",
                "Year-end store employee headcount",
                "Private label SKUs in portfolio",
                "Number of new store openings (cumulative in year)",
                "Online order fulfillment capacity (daily)"
            ],
            "Energy and Utilities": [
                "Total power generation capacity (MW)",
                "Number of power plants at year-end",
                "Number of facilities at year-end",
                "Percentage of renewable energy capacity",
                "Transmission network length",
                "Distribution network length",
                "Total number of customers connected",
                "Proven oil reserves (if applicable)",
                "Proven gas reserves (if applicable)",
                "Refinery throughput capacity",
                "Pipeline network length",
                "Greenhouse gas emissions intensity (CO₂/MWh)",
                "Year-end weighted average cost of energy production"
            ],
            "Hospitality": [
                "Number of properties at year-end",
                "Number of hotels at year-end",
                "Total number of rooms available",
                "Year-end occupancy rate",
                "Average daily rate (ADR) at final period",
                "Revenue per available room (RevPAR) at final period",
                "Loyalty program membership at year-end",
                "Number of restaurants",
                "Number of bars",
                "Conference/banquet space capacity (sq. ft.)",
                "Franchise agreements in force",
                "Hospitality workforce headcount"
            ],
            "Telecommunications": [
                "Mobile subscriber base at year-end",
                "Broadband subscriber base at year-end",
                "Mobile coverage area (population %)",
                "Mobile coverage area (geography %)",
                "Number of broadband subscribers",
                "Number of fiber subscribers",
                "Fiber network length (km)",
                "Fiber network length (miles)",
                "Average revenue per user (ARPU) at year-end",
                "5G coverage ratio (population %)",
                "Data center capacity (MW)",
                "Data center capacity (racks)",
                "Number of retail stores",
                "Number of service stores",
                "Network downtime (hours) in final reporting period"
            ],
            "Media & Entertainment": [
                "Number of streaming platform subscribers",
                "Number of online platform subscribers",
                "Broadcast coverage area (population reach)",
                "Advertising inventory at year-end",
                "Number of active licensing deals",
                "Size of film/TV content library (hours)",
                "Size of film/TV content library (titles)",
                "Social media follower count (all platforms)",
                "Year-end box office market share (if applicable)",
                "Number of production facilities",
                "In-house production capacity (titles/year)",
                "Headcount for creative roles",
                "Headcount for production roles"
            ],
            "Pharmaceuticals": [
                "Number of drugs on the market (approved)",
                "Number of compounds in Phase I",
                "Number of compounds in Phase II",
                "Number of compounds in Phase III",
                "Manufacturing capacity (units/year)",
                "Manufacturing capacity (liters/year)",
                "Global distribution network (markets served)",
                "Number of active pharmaceutical patents",
                "Clinical trial sites operating at year-end",
                "Inventory of active pharmaceutical ingredients",
                "Size of sales force (year-end)",
                "Pharmacovigilance reports (adverse events logged)",
                "Branded product count",
                "Generic product count"
            ],
            "Aerospace & Defense": [
                "Order backlog (value) at year-end",
                "Order backlog (units) at year-end",
                "Production capacity (aircraft/year)",
                "Production capacity (units/year)",
                "Number of defense contracts active",
                "Number of government contracts active",
                "R&D spending on advanced programs",
                "Number of employees with security clearance",
                "Military products in service (units)",
                "Defense products in service (units)",
                "Satellite capacity in orbit",
                "Spacecraft capacity in orbit",
                "Facilities footprint (sq. ft.)",
                "Facilities footprint (number of sites)",
                "Year-end patent portfolio (aerospace tech)",
                "Partnerships with government agencies at year-end"
            ],
            "Transport & Logistics": [
                "Fleet size (vehicles) at year-end",
                "Fleet size (aircraft) at year-end",
                "Fleet size (vessels) at year-end",
                "Warehouse capacity (sq. ft.)",
                "Warehouse capacity (cubic ft.)",
                "Number of distribution hubs",
                "Global route coverage (countries served)",
                "Global route coverage (regions served)",
                "Final-period on-time delivery rate",
                "Freight volume capacity (TEU)",
                "Freight volume capacity (tons)",
                "Fuel consumption rate (liters/year)",
                "Fuel consumption rate (per mile)",
                "CO₂ emissions from operations (ton/year)",
                "Year-end logistics staff headcount",
                "Infrastructure investments completed in the period"
            ],
            "Food & Beverage": [
                "Production capacity (e.g., bottling liters/hour)",
                "Number of manufacturing plants",
                "Number of warehouses in distribution network",
                "Number of depots in distribution network",
                "SKU count in portfolio",
                "Raw material supply contracts",
                "Inventory of raw materials at year-end",
                "Number of company-owned outlets",
                "Number of franchised outlets",
                "Year-end market share (by product category)",
                "Food safety certifications (sites certified)",
                "Brand portfolio size (distinct brands at year-end)"
            ]
        }
        return industry_metrics[industry]


if __name__ == "__main__":
    extractor = QuestionExtractor()

    # Example list of question strings (extracted from your sample JSON)
    questions = [
        "According to the annual report, what is the Operating margin (%) for Altech Chemicals Ltd  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Operating margin (%) for Cofinimmo  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "Did Cofinimmo outline any new ESG initiatives in the annual report?",
        "What is the total number of employees let go by Hagerty, Inc. according to the annual report? If data is not available, return 'N/A'.",
        "Which leadership **positions** changed at Renold plc in the reporting period? If data is not available, return 'N/A'.",
        "What was the Gross margin (%) for Charles & Colvard, Ltd. according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "What was the Capital expenditures (in GBP) for Harworth Group plc according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "What was the Capital expenditures (in USD) for Charles & Colvard, Ltd. according to the annual report (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "What are the names of new products launched by Zymeworks Inc. as mentioned in the annual report?",
        "For Lipocine Inc., what was the value of Number of diagnostic centers at year-end at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Total revenue (in USD) for Winnebago Industries, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "For Lipocine Inc., what was the value of Value of medical equipment (balance sheet) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Operating margin (%) for Audalia Resources Limited  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Total revenue (in USD) for Lipocine Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "Did Charles & Colvard, Ltd. mention any ongoing litigation or regulatory inquiries?",
        "Which leadership **positions** changed at Enerflex Ltd. in the reporting period? If data is not available, return 'N/A'.",
        "Did HV Bancorp, Inc. mention any mergers or acquisitions in the annual report?",
        "For Alien Metals Limited, what was the value of End-of-year total headcount at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "For Renold plc, what was the value of Warehouse capacity (cubic ft.) at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Operating margin (%) for Lipocine Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "According to the annual report, what is the Total revenue (in EUR) for Cofinimmo  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "Did Canadian Tire Corporation announce a share buyback plan in the annual report?",
        "Which leadership **positions** changed at Canadian Tire Corporation in the reporting period? If data is not available, return 'N/A'.",
        "Did LVMH mention any mergers or acquisitions in the annual report?",
        "For Winnebago Industries, Inc., what was the value of Number of hybrid models available at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "For Johns Lyng Group Limited, what was the value of Total expensed R&D expenditure at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "Did Cofinimmo announce any changes to its dividend policy in the annual report?",
        "What was the value of Distribution network length of Maxeon Solar Technologies, Ltd. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "What was the value of Healthcare plan memberships (if applicable) of Nevro Corp. at the end of the period listed in annual report? If data is not available, return 'N/A'.",
        "What was the largest single spending of Maxeon Solar Technologies, Ltd. on executive compensation in USD?"
    ]

    for q in questions:
        extracted = extractor.extract(q)
        # if extracted["metric"] is None or extracted["companies"] == []:
        print("Extracted Data:", extracted)
        print()

    # metrics = extractor.all_metrix()
    # print(metrics)
