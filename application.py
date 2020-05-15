from flask import Flask, session
from flask_session import Session
from flask import jsonify
from tempfile import mkdtemp
import networkx
from gensim.test.utils import get_tmpfile
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from node2vec import Node2Vec

from Step3 import Plotter


app = Flask(__name__)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.secret_key = "super secret key"

# Zvi Mints And Eilon Tsadok
#=============================================== / route ================================================#
@app.route("/")
def index():
    return "Server is UP"

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
           return jsonify(err="405", msg = "Invalid JSON file name")


    # Generate picture of networkx
    # networkx.draw(G, node_size=1)
    # plt.savefig("./load/results/networkx_before_remove.png")

    # write json formatted data
    app.logger.debug('loaded dataset with %s nodes before remove' % len(G.nodes()))
    before = json_graph.node_link_data(G)  # node-link format to serialize

    # After Remove
    for component in list(networkx.connected_components(G)):
        if len(component) <= 2: # This will actually remove only 2-connected
            for node in component:
                G.remove_node(node)

    # write json formatted data
    app.logger.debug('loaded dataset with %s nodes after remove' % len(G.nodes()))
    after = json_graph.node_link_data(G)  # node-link format to serialize

    # Save after remove graph
    networkx.write_multiline_adjlist(G, "./load/graph.adjlist")

    # Plotting
    networkx.draw(G, node_size=3)
    plt.savefig("./load/networkx_after_remove.png")

    session["load_step"] = True
    return jsonify(before=before, after=after,before_path="/load/networkx_before_remove.png", after_path="load/networkx_after_remove.png")

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
    f.close()

#=============================================== main embedding route ================================================#
@app.route("/embedding", methods=['GET'])
def embedding():
    if not "load_step" in session:
        return jsonify(err="405",msg = "Please make /load step first")

    G = networkx.read_multiline_adjlist("./load/graph.adjlist")

     # Precompute probabilities and generate walks
    node2vec = Node2Vec(G, dimensions=64, walk_length=25, num_walks=10, workers=1)
    saveWalks(list(node2vec.walks))

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Save the model into
    fname = "./embedding/test_embedded_vectors_model.kv"
    path = get_tmpfile(fname)
    model.wv.save(path)

    session["embedding_step"] = True
    return jsonify(res = "walks saved successfully" path="/embedding/test_embedded_vectors_model.kv")

#=============================================== pca route ================================================#
@app.route("/pca", methods=['GET'])
def pca():
    if not "embedding_step" in session:
        return jsonify(err="405",msg = "Please make /embedding step first")

    # Taking G from memory
    G = networkx.read_multiline_adjlist("./load/graph.adjlist")

    # Taking Memory from memory
    fname = "./embedding/test_embedded_vectors_model.kv"
    path = get_tmpfile(fname)
    model = KeyedVectors.load(path, mmap='r')

    #PCA from 64D to 3D
    plotter = Plotter.Plotter(G, model)
    plot = plotter.BaseGraph.getPlot()
    plot.savefig("./pca/BaseGraph.png")
    return jsonify(res = "pca completed and saved in image", path="/pca/BaseGraph.png")
