import logging

from flask import Flask, render_template
from flask_session import Session
from flask import jsonify
from tempfile import mkdtemp
import networkx
from networkx.readwrite import json_graph
import matplotlib
import matplotlib.pyplot as plt

app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Serve the file over http to allow for cross origin requests
app = Flask(__name__)
app.secret_key = "super secret key"

# Zvi Mints And Eilon Tsadok
@app.route("/")
def index():
    return "Server is UP"

@app.route("/graph")
def graph():
    return jsonify("this is text that i get from backend via Flask framework (here will be the graph)")

#=============================================== load route ================================================#
@app.route("/load/<string:dataset>", methods=['GET'])
def load(dataset):
    app.logger.info('got ./load request')

    # Making G (networkx)
    if dataset == "pan12-sexual-predator-identification-training-corpus-2012-05-01":
        G = networkx.read_multiline_adjlist("./load/train_networkxBeforeRemove.adjlist")
    elif dataset == "pan12-sexual-predator-identification-test-corpus-2012-05-17":
        G = networkx.read_multiline_adjlist("./load/test_networkxBeforeRemove.adjlist")
    else:
           return jsonify(err="405 Method Not Allowed - Invalid JSON file name")

    session["dataset"] = dataset
    app.logger.debug('dataset saved successfully on session')

    # Generate picture of networkx
    # networkx.draw(G, node_size=1)
    # plt.savefig("./load/results/networkx_before_remove.png")
    app.logger.debug('loaded dataset with %s nodes before remove' % len(G.nodes()))

    # write json formatted data
    before = json_graph.node_link_data(G)  # node-link format to serialize

    # After Remove
    for component in list(networkx.connected_components(G)):
        if len(component) <= 2: # This will actually remove only 2-connected
            for node in component:
                G.remove_node(node)

    app.logger.debug('loaded dataset with %s nodes after remove' % len(G.nodes()))

    # write json formatted data
    after = json_graph.node_link_data(G)  # node-link format to serialize

    # Save after remove graph
    networkx.write_multiline_adjlist(G, "./load/graph.adjlist")
    session["G"] = G
    app.logger.debug('G saved successfully on session')

    # Plotting
    networkx.draw(G, node_size=3)
    plt.savefig("./load/networkx_after_remove.png")

    return jsonify(before=before, after=after)

#=============================================== embedding route ================================================#
def saveWalks(walks):
    f = open("./embedding/walks.txt", "w+")
    row = 1
    for sentence in walks:
        f.write("row %s:    " % str(row))
        row = row + 1
        for word in sentence:
            f.write(word)
            f.write("  ")
        f.write("\n")

    session["walks"] = f
    app.logger.debug('walks saved successfully on session')
    f.close()
#=============================================== main embedding route ================================================#
@app.route("/embedding", methods=['GET'])
def embedding():
    if not "dataset" or "G" in session:
        return jsonify(err="405 Method Not Allowed - Please make /load step first")

    G = session["G"][0]
    
     # Precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=64, walk_length=25, num_walks=10, workers=1)
    saveWalks(list(node2vec.walks))

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    session["mode.wv"] = model.wv
    app.logger.debug('mode.wv saved successfully on session')

    return jsonify("model& walks saved successfully on path embedding/walks.txt")
