# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Default category definitions.
Customize these descriptions to improve classification accuracy.
"""

from categories.tree import CategoryNode, CategoryTree


def create_default_categories() -> CategoryTree:
    """
    A generic, broadly-useful category tree — sensible defaults for anyone.

    For a more specific tree, pass --categories-file, or let the LLM build one
    from your files with --discover-categories.
    """
    tree = CategoryTree()

    # ========== DOCUMENTS ==========
    documents = CategoryNode(
        name="Documents",
        description="General documents, PDFs, letters, forms, contracts, agreements, "
                    "scanned paperwork, notes, and reference material",
        threshold=0.40,
    )
    finance = documents.add_child(CategoryNode(
        name="Finance",
        description="Financial documents: bank statements, taxes, receipts, invoices, "
                    "bills, budgets, investments, payroll, insurance",
        threshold=0.50,
    ))
    finance.add_child(CategoryNode(name="Taxes",
        description="Tax returns, T4, W2, 1099, CRA, IRS, tax forms and assessments", threshold=0.55))
    finance.add_child(CategoryNode(name="Banking",
        description="Bank and credit-card statements, transactions, account summaries", threshold=0.55))
    finance.add_child(CategoryNode(name="Receipts",
        description="Receipts, invoices, purchase records, expense and order confirmations", threshold=0.55))
    documents.add_child(CategoryNode(name="Legal",
        description="Contracts, agreements, NDAs, leases, wills, legal notices and IDs", threshold=0.50))

    # ========== WORK ==========
    work = CategoryNode(
        name="Work",
        description="Professional and job-related files: business documents, reports, "
                    "presentations, meeting notes, resumes, employment, clients, projects",
        threshold=0.40,
    )
    work.add_child(CategoryNode(name="Resumes & Cover Letters",
        description="Resume, CV, cover letters, job applications, references", threshold=0.50))
    work.add_child(CategoryNode(name="Projects",
        description="Work projects, deliverables, specifications, technical documents", threshold=0.50))
    work.add_child(CategoryNode(name="Meetings",
        description="Meeting notes, agendas, minutes, action items", threshold=0.50))
    work.add_child(CategoryNode(name="Presentations",
        description="Slide decks, presentations, pitch decks, keynotes", threshold=0.50))

    # ========== SCHOOL ==========
    school = CategoryNode(
        name="School",
        description="Educational materials: homework, assignments, lecture notes, "
                    "course content, exams, quizzes, study guides, syllabi, research papers",
        threshold=0.42,
    )
    school.add_child(CategoryNode(name="Assignments",
        description="Homework, problem sets, assignments, lab reports, projects", threshold=0.50))
    school.add_child(CategoryNode(name="Notes",
        description="Lecture notes, study guides, summaries, flashcards", threshold=0.50))
    school.add_child(CategoryNode(name="Exams",
        description="Exams, tests, quizzes, midterms, finals, practice problems", threshold=0.50))
    school.add_child(CategoryNode(name="Research",
        description="Research papers, journal articles, theses, citations, references", threshold=0.50))

    # ========== MEDIA ==========
    media = CategoryNode(
        name="Media",
        description="Photos, images, screenshots, videos, music and audio files",
        threshold=0.42,
    )
    photos = media.add_child(CategoryNode(name="Photos",
        description="Photographs, pictures, selfies, vacation and family photos", threshold=0.50))
    photos.add_child(CategoryNode(name="Screenshots",
        description="Screenshots, screen captures, screen grabs", threshold=0.55))
    photos.add_child(CategoryNode(name="Camera",
        description="Camera photos, DCIM, phone pictures", threshold=0.55))
    media.add_child(CategoryNode(name="Videos",
        description="Video files, movies, recordings, clips", threshold=0.50))
    media.add_child(CategoryNode(name="Music",
        description="Music, songs, albums, audio, podcasts", threshold=0.50))

    # ========== CODE ==========
    code = CategoryNode(
        name="Code",
        description="Source code, scripts, programming projects, repositories, "
                    "configuration files, notebooks",
        threshold=0.45,
    )

    # ========== PERSONAL ==========
    personal = CategoryNode(
        name="Personal",
        description="Personal and private files: health, hobbies, travel, family, "
                    "and anything not work or school related",
        threshold=0.40,
    )
    personal.add_child(CategoryNode(name="Health",
        description="Medical records, prescriptions, doctor visits, insurance claims, fitness", threshold=0.50))
    personal.add_child(CategoryNode(name="Travel",
        description="Trips, itineraries, bookings, tickets, boarding passes, reservations", threshold=0.50))

    # ========== ARCHIVES ==========
    archives = CategoryNode(
        name="Archives",
        description="Compressed archives and installers: zip, tar, rar, 7z, dmg, iso, exe",
        threshold=0.50,
    )

    for cat in (documents, work, school, media, code, personal, archives):
        tree.add_root_category(cat)

    tree.ensure_all_other_folders()
    return tree


def create_waterloo_example_categories() -> CategoryTree:
    """Example of a highly-specific personal tree (a Waterloo CS student). Reference only."""
    tree = CategoryTree()

    # ========== SCHOOL ==========
    school = CategoryNode(
        name="School",
        description="Educational materials, homework, assignments, lecture notes, academic papers, course content, exams, quizzes, study guides, textbooks, syllabi, academic projects",
        threshold=0.45
    )
    
    # High School
    high_school = school.add_child(CategoryNode(
        name="High School",
        description="High school coursework, grades 9-12, secondary education, AP classes, IB program, high school diploma, SAT ACT prep, college applications",
        threshold=0.50
    ))
    high_school.add_child(CategoryNode(
        name="Math",
        description="High school mathematics, algebra, geometry, trigonometry, pre-calculus, calculus, math homework, math tests",
        threshold=0.55
    ))
    high_school.add_child(CategoryNode(
        name="Science",
        description="High school science, biology, chemistry, physics, lab reports, science fair, experiments",
        threshold=0.55
    ))
    high_school.add_child(CategoryNode(
        name="English",
        description="High school English, literature, essays, book reports, creative writing, grammar",
        threshold=0.55
    ))
    high_school.add_child(CategoryNode(
        name="History",
        description="High school history, social studies, world history, US history, civics, geography",
        threshold=0.55
    ))
    
    # University
    university = school.add_child(CategoryNode(
        name="University",
        description="University coursework, college, undergraduate, graduate, bachelor's degree, master's, PhD, university assignments, lectures, professors, TAs, office hours, course codes, Waterloo, UW",
        threshold=0.50
    ))
    
    # ===== CS Courses =====
    university.add_child(CategoryNode(
        name="CS 145",
        description="CS 145 Designing Functional Programs Advanced Level, functional programming, Racket, recursion, higher-order functions, Waterloo CS145",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 136L",
        description="CS 136L Tools and Techniques for Software Development, Linux, bash, git, version control, debugging, testing, Waterloo CS136L",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 146",
        description="CS 146 Elementary Algorithm Design and Data Abstraction Advanced Level, algorithms, data structures, ADTs, complexity analysis, Waterloo CS146",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 240",
        description="CS 240 Data Structures and Data Management, AVL trees, hash tables, heaps, tries, range searching, string matching, compression, Waterloo CS240",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 241",
        description="CS 241 Foundations of Sequential Programs, compilers, assemblers, MIPS, ARM assembly, sequential programming, scanning, parsing, code generation, Waterloo CS241",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 245",
        description="CS 245 Logic and Computation, propositional logic, predicate logic, formal proofs, natural deduction, Turing machines, computability, undecidability, Waterloo CS245",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 246",
        description="CS 246 Object-Oriented Software Development, C++, OOP, design patterns, RAII, inheritance, polymorphism, UML, software engineering, Waterloo CS246",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="CS 251",
        description="CS 251 Computer Organization and Design, digital logic, CPU design, pipelining, memory hierarchy, cache, assembly language, computer architecture, Waterloo CS251",
        threshold=0.60
    ))
    
    # ===== Math Courses =====
    university.add_child(CategoryNode(
        name="MATH 135",
        description="MATH 135 Algebra for Honours Mathematics, proofs, divisibility, primes, congruences, complex numbers, polynomials, Waterloo MATH135",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="MATH 136",
        description="MATH 136 Linear Algebra 1 for Honours Mathematics, vectors, matrices, linear transformations, eigenvalues, eigenvectors, Waterloo MATH136",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="MATH 137",
        description="MATH 137 Calculus 1 for Honours Mathematics, limits, derivatives, integrals, differential calculus, Waterloo MATH137",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="MATH 138",
        description="MATH 138 Calculus 2 for Honours Mathematics, integration techniques, sequences, series, Taylor series, Waterloo MATH138",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="MATH 239",
        description="MATH 239 Introduction to Combinatorics, graph theory, enumeration, generating functions, combinatorial proofs, trees, Waterloo MATH239",
        threshold=0.60
    ))
    
    # ===== Stats Courses =====
    university.add_child(CategoryNode(
        name="STAT 230",
        description="STAT 230 Probability, probability distributions, random variables, expected value, variance, binomial, Poisson, normal distribution, Waterloo STAT230",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="STAT 231",
        description="STAT 231 Statistics, statistical inference, hypothesis testing, confidence intervals, regression, estimation, Waterloo STAT231",
        threshold=0.60
    ))
    
    # ===== CO Courses =====
    university.add_child(CategoryNode(
        name="CO 250",
        description="CO 250 Introduction to Optimization, linear programming, simplex method, duality, integer programming, network flows, Waterloo CO250",
        threshold=0.60
    ))
    
    # ===== Other Courses =====
    university.add_child(CategoryNode(
        name="ECON 101",
        description="ECON 101 Introduction to Microeconomics, supply demand, markets, consumer theory, producer theory, Waterloo ECON101",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="ECON 102",
        description="ECON 102 Introduction to Macroeconomics, GDP, inflation, unemployment, monetary policy, fiscal policy, Waterloo ECON102",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="PHYS 111",
        description="PHYS 111 Physics 1, mechanics, kinematics, dynamics, energy, momentum, rotational motion, Waterloo PHYS111",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="AFM 101",
        description="AFM 101 Introduction to Financial Accounting, accounting, financial statements, balance sheet, income statement, journal entries, Waterloo AFM101",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="PSYCH 101",
        description="PSYCH 101 Introductory Psychology, psychology, behavior, cognition, perception, learning, memory, Waterloo PSYCH101",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="COMMST 223",
        description="COMMST 223 Public Speaking, communication, presentations, rhetoric, speech, Waterloo COMMST223",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="ENGL 108D",
        description="ENGL 108D Digital Lives, English, digital media, writing, communication, Waterloo ENGL108D",
        threshold=0.60
    ))
    university.add_child(CategoryNode(
        name="PD Courses",
        description="PD 1 PD 11 Professional Development, career fundamentals, technical report writing, co-op preparation, Waterloo PD courses",
        threshold=0.55
    ))
    
    tree.add_root_category(school)
    
    # ========== WORK ==========
    work = CategoryNode(
        name="Work",
        description="Professional documents, job-related files, employment, career, business, corporate, company, office, workplace, meetings, reports, presentations, co-op, internship",
        threshold=0.45
    )
    
    # ----- UBC (Past Co-op) -----
    ubc = work.add_child(CategoryNode(
        name="UBC",
        description="University of British Columbia, UBC, Department of Electrical and Computer Engineering, Vancouver, Summer Research Intern, computer vision, rehabilitation, Parkinson's, OpenCV, MediaPipe",
        threshold=0.50
    ))
    ubc.add_child(CategoryNode(
        name="Projects",
        description="UBC research projects, computer vision, rehabilitation applications, code, technical work",
        threshold=0.55
    ))
    ubc.add_child(CategoryNode(
        name="Documentation",
        description="UBC documentation, reports, research notes, technical documentation",
        threshold=0.55
    ))
    ubc.add_child(CategoryNode(
        name="Communications",
        description="UBC emails, correspondence, meeting notes, supervisor communications",
        threshold=0.55
    ))
    
    # ----- Northern Digital Inc (Upcoming Interview) -----
    ndi = work.add_child(CategoryNode(
        name="Northern Digital Inc",
        description="Northern Digital Inc, NDI, interview preparation, company research",
        threshold=0.50
    ))
    ndi.add_child(CategoryNode(
        name="Interview Prep",
        description="NDI interview preparation, technical questions, company research, interview notes",
        threshold=0.55
    ))
    ndi.add_child(CategoryNode(
        name="Projects",
        description="NDI projects, work assignments, code, technical work",
        threshold=0.55
    ))
    ndi.add_child(CategoryNode(
        name="Communications",
        description="NDI emails, correspondence, HR communications",
        threshold=0.55
    ))
    
    # ----- Template for Future Companies -----
    # (User can duplicate this pattern for new employers)
    
    # ----- Job Search (General) -----
    job_search = work.add_child(CategoryNode(
        name="Job Search",
        description="Job search, job hunting, applications, WaterlooWorks, Indeed, LinkedIn, career search, co-op search, internship applications",
        threshold=0.50
    ))
    job_search.add_child(CategoryNode(
        name="Resumes",
        description="Resume, CV, curriculum vitae, work history, professional summary",
        threshold=0.55
    ))
    job_search.add_child(CategoryNode(
        name="Cover Letters",
        description="Cover letter, application letter, letter of intent, motivation letter",
        threshold=0.55
    ))
    job_search.add_child(CategoryNode(
        name="Applications",
        description="Job application materials, application form, WaterlooWorks applications, company applications, application tracking",
        threshold=0.55
    ))
    job_search.add_child(CategoryNode(
        name="Interview Prep",
        description="General interview preparation, behavioral questions, STAR method, technical interview prep, leetcode, coding challenges",
        threshold=0.55
    ))
    job_search.add_child(CategoryNode(
        name="References",
        description="Reference letters, recommendation letters, reference contacts, reference requests",
        threshold=0.55
    ))
    
    tree.add_root_category(work)
    
    # ========== PERSONAL ==========
    personal = CategoryNode(
        name="Personal",
        description="Personal files, private documents, non-work non-school, hobbies, entertainment, lifestyle, family, friends",
        threshold=0.45
    )
    
    # Photos
    photos = personal.add_child(CategoryNode(
        name="Photos",
        description="Photographs, images, pictures, selfies, vacation photos, family photos, screenshots, camera roll",
        threshold=0.50
    ))
    photos.add_child(CategoryNode(
        name="Screenshots",
        description="Screenshots, screen captures, screen grabs",
        threshold=0.55
    ))
    photos.add_child(CategoryNode(
        name="Camera",
        description="Camera photos, DCIM, IMG_, photographs taken with phone or camera",
        threshold=0.55
    ))
    
    # Music
    personal.add_child(CategoryNode(
        name="Music",
        description="Music files, songs, albums, playlists, audio, mp3, wav, flac, spotify",
        threshold=0.50
    ))
    
    # Videos
    personal.add_child(CategoryNode(
        name="Videos",
        description="Video files, movies, recordings, clips, mp4, mkv, avi, youtube downloads",
        threshold=0.50
    ))
    
    # Finance
    finance = personal.add_child(CategoryNode(
        name="Finance",
        description="Financial documents, bank statements, taxes, receipts, invoices, budget, investments, bills, scholarships",
        threshold=0.50
    ))
    finance.add_child(CategoryNode(
        name="Taxes",
        description="Tax returns, T4, W2, tax forms, CRA, IRS, tax documents",
        threshold=0.55
    ))
    finance.add_child(CategoryNode(
        name="Banking",
        description="Bank statements, transactions, account statements, credit card statements",
        threshold=0.55
    ))
    finance.add_child(CategoryNode(
        name="Receipts",
        description="Receipts, purchase records, invoices, expense records",
        threshold=0.55
    ))
    finance.add_child(CategoryNode(
        name="Scholarships",
        description="Scholarship documents, President's Scholarship, entrance scholarship, award letters, financial aid",
        threshold=0.55
    ))
    
    # Health
    personal.add_child(CategoryNode(
        name="Health",
        description="Medical records, health documents, prescriptions, doctor visits, insurance claims, fitness",
        threshold=0.50
    ))
    
    # Gaming
    personal.add_child(CategoryNode(
        name="Gaming",
        description="Video games, game saves, gaming screenshots, mods, game configurations",
        threshold=0.50
    ))
    
    # Projects (Personal)
    personal_projects = personal.add_child(CategoryNode(
        name="Projects",
        description="Personal projects, side projects, hobby projects, My Fridge App, Android app, RoboCup, robotics",
        threshold=0.50
    ))
    personal_projects.add_child(CategoryNode(
        name="My Fridge App",
        description="My Fridge App, Android app, Google Play, fridge inventory, mobile app development",
        threshold=0.55
    ))
    personal_projects.add_child(CategoryNode(
        name="Robotics",
        description="RoboCup Junior, Maze Rescue, robotics competition, robot programming",
        threshold=0.55
    ))
    
    tree.add_root_category(personal)
    
    # Ensure all categories have "Other" subfolders
    tree.ensure_all_other_folders()
    
    return tree


def create_minimal_categories() -> CategoryTree:
    """Create a minimal category tree for testing."""
    tree = CategoryTree()
    
    school = CategoryNode(
        name="School",
        description="Educational materials, homework, assignments, lectures, academic content",
        threshold=0.45
    )
    
    work = CategoryNode(
        name="Work",
        description="Professional documents, job-related files, business, corporate",
        threshold=0.45
    )
    
    personal = CategoryNode(
        name="Personal",
        description="Personal files, hobbies, entertainment, private documents",
        threshold=0.45
    )
    
    tree.add_root_category(school)
    tree.add_root_category(work)
    tree.add_root_category(personal)
    tree.ensure_all_other_folders()
    
    return tree


def add_company(tree: CategoryTree, company_name: str, description: str) -> CategoryNode:
    """
    Helper function to add a new company to the Work category.
    
    Usage:
        tree = create_default_categories()
        add_company(tree, "Google", "Google, Alphabet, software engineering internship, Mountain View")
    """
    # Find Work category
    work_category = None
    for cat in tree.root_categories:
        if cat.name == "Work":
            work_category = cat
            break
    
    if work_category is None:
        raise ValueError("Work category not found in tree")
    
    company = CategoryNode(
        name=company_name,
        description=description,
        threshold=0.50
    )
    company.add_child(CategoryNode(
        name="Projects",
        description=f"{company_name} projects, work assignments, code, technical work",
        threshold=0.55
    ))
    company.add_child(CategoryNode(
        name="Documentation",
        description=f"{company_name} documentation, reports, technical docs",
        threshold=0.55
    ))
    company.add_child(CategoryNode(
        name="Communications",
        description=f"{company_name} emails, correspondence, meeting notes",
        threshold=0.55
    ))
    company.ensure_other_folder()
    
    # Insert before Job Search (keep Job Search at the end)
    job_search_idx = None
    for i, child in enumerate(work_category.children):
        if child.name == "Job Search":
            job_search_idx = i
            break
    
    if job_search_idx is not None:
        work_category.children.insert(job_search_idx, company)
    else:
        work_category.children.append(company)
    
    return company
