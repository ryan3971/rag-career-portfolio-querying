PROMPT_CONTEXT = """Context: This is in reference to 

"""

QUESTION_GEN_PROMPT = """Your task is to write a question given a context. Your question must be in the form of an adult mentee seeking advice 
from a trusted mentor. Formulate your question in the same style as questions users could ask in a search engine. Your question must be 
answerable with a specific, concise piece of information from the context. 

The context is below:
----------------------
{context_str}
----------------------

Your question MUST be short, clear, and based on the essence of the context. DO NOT use any qualifiers, relative clauses, or introductory modifiers.  
Keep your question short and to the point. Ask your question using the first person perspective, in the form of a student seeking advice from a trusted mentor.
"""

KEYWORD_EXTRACT_PROMPT = """Extract {keywords} exact keywords or phrases from the following text: {context_str} """

HYPE_ANSWER_GEN_PROMPT = """You're a trusted mentor to an adult mentee. Your mentee is seeking advice in the form of a question.

Below is your mentee's question:

----------------------
{query_str}
----------------------

You have some raw thoughts which you must use to formulate an answer to your mentee's question. Below are your thoughts:

----------------------
{context_str}
----------------------

Reflect on the question and your raw thoughts, then answer your mentee's question. Your response must be based on your raw thoughts, not on prior knowledge. 

DO NOT use any qualifiers, relative clauses, or introductory modifiers in your answer. Provide your answer question using the second person
perspective, speaking directly to your mentee, in the form of a OG mentor who has been there and done that and is now coming back with the
facts and giving them back to you. Use a HYPE tone and be straight up with your mentee! REMEMBER: Your response must be based on your raw thoughts, not on prior knowledge.
"""

QUERY_GEN_PROMPT = """Users aren't always the best at articulating what they're looking for. Your task is to understand the 
essense of the user query and generate {num_queries} alternate queries to expand the users query so it's more robust. This way the user will
recieve the most relevant information. 

Examples are delimited by triple backticks (```) below

````
User Query: What were some challenges faced and how were they overcome?

Alternate Queries:

1. What obstacles were encountered, and what solutions were implemented to address them?
2. What were some problems encountered, and how were they mitigated?
3. What issues surfaced during the project, and how were they handled?
````

````
User Query: When did you demonstrate acts of leadership?

Alternate Queries:

1. When have you taken on a leadership role?
2. Can you provide examples of times you led a team or project?
3. When did you step up to guide others or take charge?
4. When have you shown leadership qualities or initiative?

````
````
User Query: What technologies are you most comfortable working with?

Alternate Queries:

1. Which technologies do you feel most proficient in using?
2. What tools or platforms do you work best with?
3. Which technologies do you have the strongest expertise in?
4. What programming languages or frameworks are you most familiar with?
```

Generate {num_queries} alternate queries, one on each line, for the following user query:\n
--------------------
User Query: {query}\n
--------------------

Alternate Queries:\n
"""