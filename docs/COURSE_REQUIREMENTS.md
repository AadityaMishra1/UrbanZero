# Course Requirements — Verbatim Reference

Reproduced from `https://github.com/ncstate-csc-coursework/csc591-software-for-robots`
(private NCSU repo, accessed 2026-04-24) for self-contained reference.
Each section is a literal copy-paste from the source file noted in the
section header. If anything here conflicts with the live course repo,
**the live repo wins** — this is a frozen snapshot.

This file exists so any agent or collaborator (e.g. ClaudeCoWork) can
read the exact grading criteria and hard requirements without needing
NCSU GitHub auth.

---

## Course Identity

> # SOFTWARE FOR ROBOTICS 🤖 CSC/ECE-591, CSC-491
> ## Spring 2026
>
> * Dr. John-Paul Ore (`jwore@ncsu.edu`)
> * In-person class, EB-II 01025, M-W 3:00PM–4:15PM

---

## Project Grade Weight (from `syllabus.md`)

> ## Course Grading - 100 points
>
> | CATEGORY           | POINTS |
> |--------------------|--------|
> | Project            | 30    |
> | Workshops          | 30    |
> | Midterm            | 10    |
> | Final              | 20    |
> | Attendence         | 10    |

> ### Project
>
> Students will conduct a project related to robotic systems that
> takes real-world data as input and outputs robotic actions, creating
> software systems for advanced robotic capabilities (new), and
> projects that use open dataset are encouraged. This project can
> consist of reproducing a study (591 students only), creating a new
> tool, extending an existing tool, adapting an existing tool to a new
> domain, or similar.
>
> **The final deliverable will be a paper a 6-8 page (full pages)
> 2-column IEEE or ACM format. Also, you will put together a short
> video (1-3 minutes, 3 minutes hard max) describing your project.**
> More details on the proposal and project will be discussed in class.
> It is expected that the project paper is of sufficient quality to be
> submitted to a workshop or conference related to software
> engineering, data mining, or robotics. Students will make their
> tools and/or evaluation artifacts available on GitHub.
>
> Project deliverables are due during the last week of classes.
>
> #### Milestones and Points
>
> | PROJECT MILESTONE | POINTS|
> |-----|------|
> | Proposal | 5 |
> | Status Report Demonstrating Progress | 5 |
> | Final Paper | 10 |
> | Final Video | 10 |

---

## Project Hard Requirements (from `project/README.md`)

> # Project Ideas
>
> Please find below some suggested ideas for the semester project.
> You are not limited to these ideas, and can fomulate your own. If
> you have your own ideas, it would be good to write a brief
> description of your project idea and we can meet to discuss it.
> Teams can have up to 3 team members.
>
> ## Team Project GitHub Repo (Required)
>
> Each team will maintain a class project GitHub repository. Students
> will be graded on making regular updates to their project
> repositories. The repo must include a `README.md`. The README must
> include key deliverables for the project and which team member is
> reposible for those contributions. Every student is exptected to
> make techincal contributions to the project.
>
> ## Running ROS (Required)
>
> Every project need to run ROS as part of the project. Acceptable
> version of ROS include ROS2, ROS1, or MicroROS. You can either
> build a system from scratch, or use an existing robotic platform an
> modify it (see suggested platforms below).
>
> ## Sensing and Actuating (Required)
>
> Every project needs to have a system that both senses and actuates.
>
> 2025 Project Checklist:
> ---
> 1. Pick a domain (i.e. Underwater welding, space station, home or
>    eldercare, **driving**, emergency reponse, volcano exploration)
> 2. Get a robot running in simulation. Characterize simulation data.
>    Make the robot do something.
> 3. Determine your task. Define what the robot will sense and how it
>    will actuate.
>
> You may also define your own project, but somewhere the system must
> run ROS.
>
> Suggested Platforms:
> ---
> * [Clearpath Jackal](https://github.com/jackal)
> * [Stanford Quadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)
> * [ROS Industrial Arms](https://github.com/orgs/ros-industrial/repositories)
> * [Autoware](https://autoware.org/)
> * [ApolloAuto](https://github.com/ApolloAuto/apollo)
> * [Linorobot](https://github.com/linorobot/linorobot2)
> * [Facebook Research Home Robot](https://github.com/facebookresearch/home-robot)
> * [Astrobee](https://github.com/nasa/astrobee) (mostly ROS 1)

### Compliance status for UrbanZero

| Requirement | Status |
|---|---|
| GitHub repo with README listing deliverables | ✅ This repo + README.md |
| Running ROS | ✅ `ros/urbanzero_node.py` (ROS 2 Humble), launched in `run.sh` Pane 2 |
| Sensing and Actuating | ✅ Semantic-seg camera in, throttle/steer out |
| Domain pick | ✅ Driving (explicitly listed in suggested domains) |
| Solo team | ✅ Allowed (teams can be 1-3) |

---

## Final Paper Rubric (verbatim from `project/project_report_rubric_final.md`)

> | ITEM | POINTS | ACTUAL |
> |--|--|--|
> | Abstract conveys the project | 0.667 | |
> | Introduction motivates the problem and identifies contributions | 1 | |
> | Relevant citations and references provide | 0.667 | |
> | Use of Figures to convey key concepts / results | 0.5 | |
> | Technical approach is explained clearly  | 0.5 | |
> | Claims supported by evidence | 0.667 | |
> | Code / Artifacts demonstrate effort | 3.5 | |
> | Related Work | 0.332 | |
> | Lesson's learned documents the journey | 1 | |
> | Polish, Spelling, Grammar | 0.5 | |
> | Figure / Graph Axis labels | 0.332 | |
> | Conclusion summarizes work | 0.332 | |
>
> Multi-Person Teams: Every Person on the team must individually
> identify their unique contributions to the project.

**Total: 10 points.** This project is solo; the multi-person
contributions clause does not apply.

---

## Final Video Rubric (verbatim from `project/final_project_presentation_rubric.md`)

> | ITEM | POINTS | ACTUAL |
> |--|--|--|
> | Introduction motivates the problem and identifies contributions | 2 | |
> | Presentation is <= 3 minute time limit | 2 | |
> | Lesson's learned and documentation of the journey | 3 | |
> | Polish, Spelling, Grammar | 1 | |
> | Conclusion summarizes work | 2 | |

**Total: 10 points.** Hard cap at 3 minutes — TAs may not grade past
the limit.

---

## Submission Mechanics (from `syllabus.md` schedule + class README)

> | DATE | DUE |
> |------|-----|
> | Apr 26 | Project and Video (3 min max for all team sizes) due at 11:59pm AOE. |

- **AOE** = Anywhere on Earth (UTC−12). Apr 26 11:59pm AOE
  ≈ Apr 27 11:59am UTC ≈ Apr 27 7:59am EDT.
- Paper submission form:
  <https://docs.google.com/forms/d/e/1FAIpQLSdofyQAXAuFc6k4HEDeC-ZiKk4rLVgwYqHm9JTZ5nEvqLjRoQ/viewform>
- Video submission form:
  <https://docs.google.com/forms/d/e/1FAIpQLSeZhx701WOPGoqqZKvjHZVOB6Ta-tN5zo8kdezFKbLMooxGgw/viewform>

---

## Status Report Guidelines (already submitted, for reference)

From `project/status-report-guidelines.md`:

> The goal is to communicate your accomplishments on the project so
> far. Please include screen capture including running code / URDF
> models / simulations. You can also discuss what you have tried, what
> has worked, and what has not worked.
>
> * Summarize. "The problem is...   It applies to...  For example..."
> * Progress: Talk about what you've done so far and how it relates
>   to your timeline.
> * Length: 3-5 minutes. Each person much explain their contributions
>   for at least 1 minute. A team of 2 will have a timing guideline
>   of 4 minutes, a team of 3 has a timing guideline of 5 minutes.
>   The 5 minute limit *will be strict* (The TAs might not grade past
>   the first 5 minutes), so practice your timing.
> * Make sure every team member contributes meaningfully.
> * Discuss next steps.

From `project/status-report-guidelines-2.md`:

> ## 4-MINUTE PROJECT STATUS REPORT GUIDELINES
>
> * (0.5 points) Use 4 minutes or less—communicate clearly, practice,
>   have a plan on how you will share your update.
> * (0.5 points) Every team member participates and identifies their
>   __individual contribution and responsibilities__.
> * (1 point) Show your project progress (code, documentation,
>   initial writeup, etc)

---

## Proposal Template (already submitted, for reference)

From `project/project-proposal-template.md`:

> ## TEAM / PROJECT NAME:
>
> ### SUMMARY:   "What are we doing?"    (1 paragraph)
>
> ### MOTIVATION:  "Who cares?" and "How could this be valuable or
> useful?"  (1-2 paragraphs)
>
> ### CHALLENGE:  "Why is this hard?"  (1 sentence to 1 paragraph)
>
> ### EXPECTED APPROACH:  "How will we do it?"  (1-2 paragraph)
>
> ### RELATED WORK:  "What else has been done?"  (1-2 paragraph)
>
> ### MILESTONES / TIMELINE /:  (3-10 items)
>
> ### MATERIALS NEEDED:   list of software/computation/hardware
>
> ### REFERENCES:   citations in IEEE/ACM format
>
> ## Evaluation
>
> This Project Proposal is worth 5 points.
>
> | ITEM | POINTS |
> |--|--|
> | Summary.  Clear and brief. | 0.75 |
> | Motivation. Connect project to something people care about (or should care about). | 0.75 |
> | Challenge. Explain why this is hard. | 0.75 |
> | Expected Approach. Describe how you *think* you'll do it. | 0.75 |
> | Milestones. Clear dates with sufficient detail to know that you've reached that goal. | 0.75 |
> | Related Work / References (at least 2) | 0.75 |
> | Submitting on time | 0.5 |

---

## Office Hours / Contact (from class README)

| Teaching Staff | Email | Office Hours |
|---|---|---|
| Adnan | `ajaljul@ncsu.edu` | Tuesdays 11:00 am - 12:00 pm Zoom |
| Dhruva | `dungrup@ncsu.edu` | Wednesdays 1:00 pm - 2:00 pm Zoom |
| Dr. Ore | `jwore@ncsu.edu` | Thursdays 10:00 am - 11:00 am Zoom |

Q&A: EdStem at `https://edstem.org/us/courses/92658`.
