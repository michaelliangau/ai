import os


# List of company descriptions
company_descriptions = [
    "We are a leading player in the biotech industry. We specialize in genetic engineering and have made significant contributions to the field of gene therapy. Our innovative solutions have helped in the treatment of various genetic disorders.",
    "We are known for our work in the field of molecular biology. We have developed a range of products that have revolutionized the field of diagnostics. Our products are used in labs across the world.",
    "We are a pioneer in the field of biopharmaceuticals. We have developed a range of drugs that have proven to be effective in the treatment of various diseases.",
    "We are a leading player in the space tech industry. We have developed a range of satellites that are used for various purposes like weather forecasting, communication, and research.",
    "We are known for our innovative solutions in the field of space exploration. Our spacecrafts have made significant contributions to the understanding of our solar system.",
    "We are a pioneer in the field of satellite technology. Our satellites are used for various purposes like weather forecasting, communication, and research.",
    "We are a leading player in the AI industry. We specialize in machine learning and have developed a range of products that have revolutionized the field of data analysis.",
    "We are known for our work in the field of artificial intelligence. Our AI models have made significant contributions to the understanding of complex data patterns.",
    "We are a pioneer in the field of AI. Our AI models are used for various purposes like data analysis, prediction, and automation."
]

if not os.path.exists('company_data'):
    os.makedirs('company_data')

company_types = ['biotech', 'biotech', 'biotech', 'space', 'space', 'space', 'ai', 'ai', 'ai']
for i, description in enumerate(company_descriptions):

    with open(f'./company_data/{company_types[i]}_{i}.txt', 'w') as f:
        f.write(description)

# List of target categories
category_descriptions = [
    # Biotechnology
    """
    Biotechnology is an interdisciplinary field that combines biological sciences with engineering and technological principles to create innovative solutions across a wide range of applications. This field has evolved significantly since its inception, with roots tracing back to ancient times when humans first began domesticating crops and animals, leading to selective breeding practices.

    In the modern era, biotechnology encompasses several key areas, including medical biotechnology, agricultural biotechnology, and environmental biotechnology. Medical biotechnology focuses on developing pharmaceuticals, gene therapies, and regenerative medicine. This includes the creation of vaccines, antibiotics, and the use of genetic engineering to treat diseases at a molecular level. For example, the development of CRISPR-Cas9 gene-editing technology has revolutionized the potential for treating genetic disorders.

    Agricultural biotechnology involves the genetic manipulation of crops to increase yield, nutritional value, and resistance to pests and diseases. This includes the development of genetically modified organisms (GMOs), which have sparked both interest and controversy regarding their safety and impact on biodiversity.

    Environmental biotechnology applies to the use of living organisms, or their products, for environmental restoration and sustainable practices. This includes bio-remediation, where microorganisms are used to clean up oil spills or heavy metal contamination, and the development of biofuels as renewable energy sources.

    The ethical implications of biotechnology are significant, including concerns over GMOs, gene editing in humans, and the impact on natural ecosystems. The field must navigate these issues while pushing the boundaries of scientific innovation.
    """,

    # Space Technology
    """
    Space technology encompasses the tools, processes, and knowledge necessary for activities related to outer space. Its history began in the mid-20th century with the space race, primarily between the United States and the Soviet Union, leading to significant milestones such as the launch of the first artificial satellite, Sputnik, and the first human landing on the Moon.

    Today, space technology includes satellite technology, which is integral to global communication systems, GPS navigation, and earth observation. Satellites provide crucial data for weather forecasting, climate monitoring, and disaster management. They also play a vital role in telecommunications, enabling global connectivity.

    Space exploration continues to be a major focus, with missions to Mars, the establishment of the International Space Station (ISS), and plans for manned missions to other planets. This area of space technology is pushing the boundaries of human knowledge and capabilities in space travel.

    The commercialization of space technology has led to the emergence of private companies like SpaceX and Blue Origin, changing the landscape of space exploration and travel. These companies are working towards making space travel more accessible and cost-effective.

    Future prospects in space technology include advancements in propulsion systems, the potential colonization of other planets, and the continued exploration of the solar system and beyond. The field also faces challenges, such as the issue of space debris and the ethical implications of space colonization.
    """,

    # Artificial Intelligence
    """
    Artificial Intelligence (AI) is a branch of computer science focused on creating systems capable of performing tasks that typically require human intelligence. This includes activities such as learning, reasoning, problem-solving, perception, and language understanding.

    The history of AI dates back to the mid-20th century, with foundational work by pioneers like Alan Turing and John McCarthy. Initially, AI research focused on symbolic approaches, trying to encode human knowledge and reasoning in a computer-readable form. However, the advent of machine learning and deep learning has shifted the focus towards creating algorithms that can learn from data, leading to significant advancements.

    AI applications are diverse and widespread, including natural language processing (NLP), computer vision, robotics, and predictive analytics. In NLP, AI is used for language translation, chatbots, and voice assistants. Computer vision allows machines to interpret and make decisions based on visual data, used in applications like autonomous vehicles and facial recognition systems.

    In recent years, AI has seen rapid growth due to increased computational power, availability of large datasets, and advancements in algorithms. This growth has led to transformative impacts across various sectors including healthcare, finance, and manufacturing.

    The future of AI holds both exciting possibilities and significant challenges. Key areas of future development include improving AI's ability to understand and interact in natural, human-like ways, ensuring ethical use of AI, and addressing concerns like job displacement and privacy.

    AI's ethical considerations are particularly critical, focusing on issues like algorithmic bias, transparency, and the potential for misuse. The field is also grappling with the challenges of AI alignment – ensuring that AI systems' goals are aligned with human values and well-being.
    """
]


if not os.path.exists('category_data'):
    os.makedirs('category_data')

category_types = ['biotech', 'space', 'ai']
for i, description in enumerate(category_descriptions):

    with open(f'./category_data/{category_types[i]}_{i}.txt', 'w') as f:
        f.write(description)
