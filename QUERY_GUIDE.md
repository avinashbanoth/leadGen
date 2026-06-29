# Query Guide — How to Get Better Lead Results

This system finds verified contact details (email, LinkedIn) of decision makers at companies.
The more specific your query, the better the results.

---

## The Three Building Blocks of a Good Query

Every effective query has three parts:

| Part | What it means | Example |
|---|---|---|
| **Role** | Who you want to reach | CTO, VP Sales, Founder, HR Head |
| **Industry** | What type of company | fintech, logistics, SaaS, PLM software |
| **Location** | Where the company operates | Bangalore, Germany, USA, Singapore |

**Minimum viable query:** at least Industry + Location.
If you skip the Role, the system will infer the most relevant decision maker for that industry.

---

## Query Templates

### Find decision makers at a category of companies

```
Find [Role] at [Industry] companies in [Location]
```

Examples:
- `Find CTOs at fintech startups in Bangalore`
- `Find VP Sales at SaaS companies in Germany`
- `Find HR heads at logistics companies in the UK with 200+ employees`
- `Find founders of e-commerce companies in Singapore`

---

### Find people at a specific named company

```
Who is the [Role] at [Company Name]
Find [Role] at [Company Name]
```

Examples:
- `Who is the CTO at Razorpay`
- `Find the VP Engineering at Zerodha`
- `Get the Head of Sales at SAP Germany`

---

### Find companies + contacts based on a business signal

```
Find [Role] at [Industry] companies in [Location] that are [signal]
```

Examples:
- `Find CTOs at fintech companies in Bangalore that recently raised Series B`
- `Find DevOps leads at SaaS companies in the US struggling with Kubernetes costs`
- `VP Engineering at startups in Germany that are hiring cloud engineers`
- `Find founders of e-commerce companies struggling with payment integration`

Signals that work well:
- `recently funded` / `Series A/B/C`
- `hiring [role/skill]`
- `struggling with [problem]`
- `using [technology]` (e.g., using AWS, using Salesforce)
- `recently launched` / `growing fast`

---

### Skip the role — let the system choose

If you don't know which role to target, omit it. The system will find the most relevant decision maker for the industry.

```
Find decision makers at [Industry] companies in [Location]
```

Examples:
- `Find decision makers at PLM software companies in Germany`
- `Fintech startups in Bangalore — get me someone to reach out to`
- `Top executives at logistics companies in the Netherlands`

---

## Role Reference — What You Can Ask For

You don't need to use exact titles. These all work:

| What you type | What the system searches for |
|---|---|
| `CTO` | CTO, Chief Technology Officer, VP Engineering, Head of Technology |
| `CEO` | CEO, Chief Executive Officer, Founder & CEO, Managing Director |
| `founder` | Founder, Co-founder, CEO, Founder & CEO |
| `VP Sales` | VP Sales, Vice President Sales, Head of Sales, Sales Director, CRO |
| `VP Engineering` | VP Engineering, Head of Engineering, Director of Engineering |
| `HR head` | HR Head, CHRO, VP People, Head of HR, HR Director |
| `CFO` | CFO, Chief Financial Officer, VP Finance, Finance Director |
| `COO` | COO, Chief Operating Officer, VP Operations |
| `CISO` | CISO, Chief Information Security Officer, VP Security, Head of Cybersecurity |
| `product manager` | Product Manager, VP Product, Head of Product, CPO |
| `devops` | DevOps Lead, Head of DevOps, SRE Lead, Platform Engineering Lead |

For roles not in the list above, just describe them naturally:
- `Head of Procurement`
- `Chief Data Officer`
- `VP of Customer Success`

---

## Company Size Filters

Add size constraints directly to your query:

| Filter | Example phrasing |
|---|---|
| Startups only | `fintech startups`, `early-stage`, `seed-stage` |
| Mid-market | `200+ employees`, `mid-size`, `Series B` |
| Enterprise | `enterprise`, `large companies`, `Fortune 500` |

Examples:
- `Find CTOs at fintech startups (not large enterprises) in Bangalore`
- `VP Engineering at SaaS companies with 200–1000 employees in Germany`
- `Founders of pre-Series A e-commerce companies in Singapore`

---

## What the System Returns

For each lead, you get:
- **Name** and **Job Title**
- **Email** — verified (✓), partial/permuted (~), or not found (✗)
- **Confidence %** — how certain the system is about the email
- **Lead Score** — 0–100 relevance score
- **LinkedIn URL** — when email isn't available
- **Company** — with website and industry

Results are capped at **5 contacts** by default to conserve API quota. Ask for more if needed.

---

## Two-Tier Results — What [L2 fallback] Means

The system first searches for C-suite and VP-level people (Level 1). If none are found at a company, it automatically falls back to Director and Manager-level contacts (Level 2) and marks them **[L2 fallback]**.

Level 2 contacts are still decision-relevant — they're the most senior person the system could find.

---

## How the System Finds People

The pipeline uses three layers in order, stopping when results are found:

1. **Apollo.io** — searches a database of 270M+ professionals by company and title. Covers most large and mid-market companies globally.
2. **Website team pages** — scrapes the company's own `/team`, `/about`, and `/leadership` pages. Works for smaller companies and European SMEs that publish their leadership online.
3. **Google dorks** — searches `site:linkedin.com/in "title" "company"` via SearXNG. Used as a last resort for large companies with heavily indexed LinkedIn profiles.

---

## What Will NOT Work

| Query type | Why it fails | Alternative |
|---|---|---|
| `Find me a job as a developer` | Job search, not lead gen | — |
| `Write a sales email` | Content task | — |
| `Find freelance designers` | Freelancer search, not B2B | — |
| `What is the weather in Berlin` | Not lead gen | — |
| `Find everyone at Google` | Too broad, no role signal | `Find VP Engineering at Google` |

---

## Vague Queries — What Happens

If your query is too vague, the system will ask a follow-up question. Example:

> You: `find startup contacts`
>
> System: *You're looking for contacts at startups — here's what I still need:*
> - *Industry (e.g. fintech, SaaS, healthcare, logistics)*
> - *Role (e.g. CTO, VP Sales, Founder — or skip and I'll target top executives)*
> - *Location (e.g. Bangalore, Germany, US)*
>
> *Try: "Find CTOs at fintech startups in Bangalore"*

---

## Quick Examples — Copy and Adapt

```
Find CTOs at fintech startups in Bangalore
Find VP Sales at logistics companies in Germany with 200+ employees
Who is the CTO at Razorpay
Find founders of e-commerce companies in Singapore struggling with payment integration
HR heads at healthcare companies in the UK
Find decision makers at PLM software companies in Germany
VP Engineering at SaaS companies in the US that recently raised Series B
Find the head of procurement at manufacturing companies in the Netherlands
CISOs at financial services companies in New York
Top executives at cloud infrastructure startups in Bangalore
```
