from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import requests
import json
import time
import re
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import os
import networkx as nx
import heapq

app = Flask(__name__)

# Function to get embeddings from the API
def get_embedding(text):
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"model": "llama3.1:8b", "input": text})
    response = requests.post('http://localhost:11434/api/embed', headers=headers, data=data)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    response_data = response.json()
    
    if 'embedding' in response_data:
        return np.array(response_data['embedding'], dtype=np.float32)
    elif 'embeddings' in response_data and response_data['embeddings']:
        return np.array(response_data['embeddings'][0], dtype=np.float32)
    else:
        raise KeyError(f"No embedding found in API response. Response: {response_data}")

# Database functions
def create_database():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB, is_question INTEGER)''')
    conn.commit()
    return conn

def insert_data(conn, text, embedding, is_question):
    c = conn.cursor()
    c.execute("INSERT INTO embeddings (text, embedding, is_question) VALUES (?, ?, ?)",
              (text, sqlite3.Binary(np.array(embedding).tobytes()), is_question))
    conn.commit()

# Annoy index functions
def build_annoy_index(conn, vector_size=4096, n_trees=10):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM embeddings")
    total_vectors = c.fetchone()[0]
    
    annoy_index = AnnoyIndex(vector_size, 'angular')
    c.execute("SELECT id, embedding FROM embeddings")
    
    for i, (id, embedding_blob) in enumerate(c.fetchall()):
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        if len(embedding) != vector_size:
            print(f"Warning: Embedding size mismatch. Expected {vector_size}, got {len(embedding)}. Skipping this vector.")
            continue
        annoy_index.add_item(id - 1, embedding)
    
    print("Building index...")
    annoy_index.build(n_trees)
    annoy_index.save('embeddings.ann')
    print("Index built and saved")

def find_similar(conn, query_embedding, top_k=5):
    annoy_index = AnnoyIndex(4096, 'angular')
    annoy_index.load('embeddings.ann')
    
    similar_ids, distances = annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    
    c = conn.cursor()
    results = []
    for id, distance in zip(similar_ids, distances):
        c.execute("SELECT text, is_question FROM embeddings WHERE id = ?", (id + 1,))
        text, is_question = c.fetchone()
        similarity = 1 - distance
        results.append((id + 1, text, similarity, bool(is_question)))
    
    return results

# Llama model interaction functions
def stream_api_call(messages, max_tokens):
    prompt = json.dumps(messages)
    data = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True
    }
    try:
        response = requests.post('http://localhost:11434/api/generate', 
                                 headers={'Content-Type': 'application/json'}, 
                                 data=json.dumps(data),
                                 stream=True)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    full_response += chunk['response'].replace("'",'')
                    yield chunk['response'].replace("'",'')
        if full_response:
            return json.loads(full_response.replace("'",''))
        else:
            raise ValueError("Empty response from API")
    except Exception as e:
        error_message = f"Failed to generate response. Error: {str(e)}"
        return {"title": "Error", "content": error_message, "next_action": "final_answer"}

def extract_json(text):
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()
    json_objects = re.findall(r'\{[^{}]*\}', text)
    
    if json_objects:
        try:
            return json.loads(json_objects[-1])
        except json.JSONDecodeError:
            pass
    
    return {
        "title": "Parsing Error",
        "content": text,
        "next_action": "continue"
    }

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def get_short_title(content):
    messages = [
        {"role": "system", "content": "You are a concise summarizer. Provide a very short title (under 20 characters) for the given content."},
        {"role": "user", "content": f"Summarize this in under 20 characters: {content[:100]}..."}
    ]
    
    title_data = ""
    for chunk in stream_api_call(messages, 50):
        title_data += chunk
    
    short_title = title_data.strip()[:20]
    return short_title

def calculate_strongest_path(graph_data, current_step):
    G = nx.Graph()
    for node in graph_data['nodes']:
        G.add_node(node['id'])
    for edge in graph_data['edges']:
        G.add_edge(edge['from'], edge['to'], weight=edge['value'])
    
    start_node = 'Step1'
    end_node = f'Step{current_step}'
    
    def dijkstra(graph, start, end):
        queue = [(0, start, [])]
        visited = set()
        
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node not in visited:
                visited.add(node)
                path = path + [node]
                
                if node == end:
                    path_length = len(path) - 1
                    if path_length == 0:  # Handle the case when there's only one node
                        return 1.0, path  # Return perfect similarity for single node
                    return -cost / path_length, path  # Return average similarity
                
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        edge_weight = graph[node][neighbor]['weight']
                        new_cost = cost - edge_weight  # Accumulate total similarity
                        heapq.heappush(queue, (new_cost, neighbor, path))
        
        return None, None

    try:
        avg_similarity, path = dijkstra(G, start_node, end_node)
        if path:
            if len(path) == 1:  # Handle the case when there's only one node
                return path, [], 1.0
            path_edges = list(zip(path[:-1], path[1:]))
            path_weights = [G[u][v]['weight'] for u, v in path_edges]
            return path, path_weights, avg_similarity
        else:
            return None, None, None
    except nx.NetworkXNoPath:
        return None, None, None

def generate_response(prompt, conn):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    graph_data = {
        'nodes': [],
        'edges': []
    }
    embeddings = []
    edge_dict = {}  # New dictionary to keep track of edges

    def serialize_graph_data(graph_data):
        serialized = {
            'nodes': graph_data['nodes'],
            'edges': [
                {
                    'from': edge['from'],
                    'to': edge['to'],
                    'value': float(edge['value']),  # Convert float32 to regular float
                    'label': f"{float(edge['value']):.2f}",  # Add similarity value as label
                    'font': {'size': 10}  # Adjust font size for readability
                }
                for edge in graph_data['edges']
            ]
        }
        #print("serialized",serialized)
        return serialized

    def calculate_top_similarities(embeddings, current_step, top_k=2):
        similarities = []
        for i in range(min(current_step, len(embeddings))):
            if i < len(embeddings) and current_step < len(embeddings):
                similarity = float(calculate_similarity(embeddings[current_step], embeddings[i]))
                similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    max_steps = 20  # Set a maximum number of steps to prevent infinite loops
    final_answer = None  # Initialize final_answer

    while step_count < max_steps:
        start_time = time.time()
        step_data = ""
        for chunk in stream_api_call(messages, 300):
            step_data += chunk
        end_time = time.time()
        thinking_time = end_time - start_time
        
        step_json = extract_json(step_data)
        title = step_json.get('title', '')
        content = step_json.get('content', 'No content')
        next_action = step_json.get('next_action', 'continue')
        
        # Check if content exceeds 700 characters
        if len(content) > 700:
            print(f"Step {step_count} content exceeded 700 characters. Retrying...")
            messages.append({"role": "user", "content": "Your last response was too long. Please provide a more concise version of your last step."})
            continue  # Skip the rest of the loop and try again
        
        # If we reach here, the step is valid and under 700 characters
        total_thinking_time += thinking_time
        
        # Calculate embedding for the current step
        embedding = get_embedding(content)
        embeddings.append(embedding)
        insert_data(conn, content, embedding, False)
        
        # Generate a short title only if the original title is empty or too long
        if not title or len(title) > 20:
            short_title = get_short_title(content)
        else:
            short_title = title[:20]  # Truncate the original title if it's longer than 20 characters
        
        # Generate a unique node ID
        node_id = f"Step{step_count}"
        while node_id in [node['id'] for node in graph_data['nodes']]:
            step_count += 1
            node_id = f"Step{step_count}"
        
        # Add node for this step
        graph_data['nodes'].append({
            'id': node_id,
            'label': f"Step {step_count}: {short_title}"
        })
        
        if step_count > 1 and len(embeddings) > 1:
            top_similarities = calculate_top_similarities(embeddings, len(embeddings) - 1, top_k=2)
            
            # Clear previous edges for the current step
            edge_dict = {k: v for k, v in edge_dict.items() if v['to'] != node_id}
            
            for prev_step, similarity in top_similarities:
                prev_node_id = f"Step{prev_step + 1}"
                if prev_node_id in [node['id'] for node in graph_data['nodes']]:  # Only create edges to existing nodes
                    edge_key = f"{prev_node_id}-{node_id}"
                    edge_dict[edge_key] = {
                        'from': prev_node_id,
                        'to': node_id,
                        'value': similarity,
                        'length': 300 * (1 - similarity)
                    }

        # Update graph_data['edges'] with the current edge_dict
        graph_data['edges'] = list(edge_dict.values())
        # Scale node sizes based on average similarity
        connected_similarities = [edge['value'] for edge in edge_dict.values() if edge['from'] == node_id or edge['to'] == node_id]
        if connected_similarities:
            avg_similarity = sum(connected_similarities) / len(connected_similarities)
            graph_data['nodes'][-1]['value'] = avg_similarity * 30 + 10  # Scale to 10-40 range
        else:
            graph_data['nodes'][-1]['value'] = 20  # Set a default size if no connections

        serialized_graph_data = serialize_graph_data(graph_data)
        strongest_path, path_weights, avg_similarity = calculate_strongest_path(serialized_graph_data, step_count)
        
        path_data = {
            'strongest_path': strongest_path,
            'path_weights': path_weights,
            'avg_similarity': avg_similarity
        } if strongest_path is not None else None

        yield f"data: {json.dumps({'type': 'step', 'step': step_count, 'title': title, 'content': content, 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
        
        steps.append((f"Step {step_count}: {title}", content, thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_json)})
        
        if next_action == 'final_answer' and step_count <= 5:
            print("Final answer requested but not enough steps provided. Continuing...")
            messages.append({
                "role": "user",
                "content": f"You've only provided {step_count - 1} steps of 5. Can you look for possible error or alternatives to your answer. Continue your reasoning."
            })
            continue
        elif next_action == 'final_answer' or 'boxed' in content.lower():
            if not final_answer:
                final_answer = content  # Set final_answer if not already set

            # Add last evaluation step
            messages.append({
                "role": "user",
                "content": f"Let's do a final evaluation. The original question was: '{prompt}'. Based on your reasoning, is your final answer correct and complete? If not, what might be missing or incorrect?"
            })
            
            start_time = time.time()
            evaluation_data = ""
            for chunk in stream_api_call(messages, 300):
                evaluation_data += chunk
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time
            
            evaluation_json = extract_json(evaluation_data)
            evaluation_content = evaluation_json.get('content', 'No evaluation content')
            
            # Check if the evaluation suggests a different answer
            if check_consistency(final_answer, evaluation_content):
                break  # Exit the loop if consistent
            else:
                print("Inconsistency detected. Restarting the reasoning process.")
                yield f"data: {json.dumps({'type': 'inconsistency', 'message': 'Inconsistency detected. Restarting the reasoning process.'})}\n\n"
                messages = messages[:2]  # Reset messages to initial state
                step_count += 1  # Increment step count instead of resetting
                final_answer = None  # Reset final_answer
                graph_data = {'nodes': [], 'edges': []}  # Reset graph data
                embeddings = []
                edge_dict = {}
                continue

        step_count += 1  # Increment step count only for valid steps

    # Generate final answer if not already provided
    if not final_answer:
        messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
        
        start_time = time.time()
        final_data = ""
        for chunk in stream_api_call(messages, 200):
            final_data += chunk
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        final_json = extract_json(final_data)
        final_answer = final_json.get('content', final_data)

    # Calculate embedding for the final answer
    final_embedding = get_embedding(final_answer)
    insert_data(conn, final_answer, final_embedding, False)
    
    # Add final answer node to the graph
    final_node_id = f"Step{step_count}"
    while final_node_id in [node['id'] for node in graph_data['nodes']]:
        step_count += 1
        final_node_id = f"Step{step_count}"
    
    graph_data['nodes'].append({
        'id': final_node_id,
        'label': f"Final Answer: {get_short_title(final_answer)}"
    })
    
    # Calculate similarities with previous steps for the final answer
    top_similarities = calculate_top_similarities(embeddings + [final_embedding], step_count - 1, top_k=2)
    
    for prev_step, similarity in top_similarities:
        prev_node_id = f"Step{prev_step + 1}"
        if prev_node_id in [node['id'] for node in graph_data['nodes']]:  # Only create edges to existing nodes
            edge_key = f"{final_node_id}-{prev_node_id}"
            edge_dict[edge_key] = {
                'from': final_node_id,
                'to': prev_node_id,
                'value': similarity,
                'length': 300 * (1 - similarity)
            }
    
    graph_data['edges'] = list(edge_dict.values())

    serialized_graph_data = serialize_graph_data(graph_data)
    strongest_path, path_weights, avg_similarity = calculate_strongest_path(serialized_graph_data, step_count)
    
    path_data = {
        'strongest_path': strongest_path,
        'path_weights': path_weights,
        'avg_similarity': avg_similarity
    } if strongest_path is not None else None

    yield f"data: {json.dumps({'type': 'final', 'content': final_answer, 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
    
    steps.append(("Final Answer", final_answer, thinking_time))

    yield f"data: {json.dumps({'type': 'done', 'total_time': total_thinking_time})}\n\n"

    # Stop processing here
    return

def clear_database(conn):
    c = conn.cursor()
    c.execute("DELETE FROM embeddings")
    conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        user_query = request.json['query']
    else:  # GET
        user_query = request.args.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    conn = create_database()

    # Clear the database before processing the new query
    clear_database(conn)

    # Add user query to database
    query_embedding = get_embedding(user_query)
    insert_data(conn, user_query, query_embedding, True)

    def generate():
        yield from generate_response(user_query, conn)

        # Rebuild Annoy index after adding new data
        build_annoy_index(conn)

        # Find similar questions/answers
        similar_items = find_similar(conn, query_embedding, top_k=5)
        yield f"data: {json.dumps({'type': 'similar', 'items': similar_items})}\n\n"

        conn.close()

    return Response(generate(), mimetype='text/event-stream')

def check_consistency(final_answer, evaluation):
    #messages = [
    #    {"role": "system", "content": "You are a consistency checker. Compare the final answer and the evaluation, and determine if they are consistent or if the evaluation suggests a significantly different answer."},
    #    {"role": "user", "content": f"Final answer: {final_answer}\n\nEvaluation: {evaluation}\n\nAre these consistent? Respond with ONLY 'consistent' or 'inconsistent'."}
    #]
    #
    #for attempt in range(5):  # Try up to 5 times
    #    response = ""
    #    for chunk in stream_api_call(messages, 50):
    #        response += chunk
    #    
    #    response = response.strip().lower()
    #    print(f"check_consistency response (attempt {attempt + 1}):", response)
    #    
    #    if response.startswith("consistent") or response.startswith("inconsistent"):
    #        return response.startswith("consistent")
    #    
    #    # If we reach here, the response was invalid, so we'll try again
    #    messages.append({"role": "user", "content": "Please respond with 'consistent' or 'inconsistent' at the beginning."})
    #
    ## If we've tried 5 times and still haven't got a valid response, default to inconsistent
    #print("Failed to get a valid consistency check after 5 attempts. Defaulting to inconsistent.")
    #return False
    return True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)