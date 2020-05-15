import logging

from threading import Thread

from flask import Flask, render_template
from flask import jsonify

from networkx.readwrite import json_graph

import networkx
import matplotlib
import matplotlib.pyplot as plt

# Serve the file over http to allow for cross origin requests
app = Flask(__name__)

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

    #making G (networkx)
    if dataset == "pan12-sexual-predator-identification-training-corpus-2012-05-01":
        G = networkx.read_multiline_adjlist("./load/train_networkxBeforeRemove.adjlist")
    elif dataset == "pan12-sexual-predator-identification-test-corpus-2012-05-17":
        G = networkx.read_multiline_adjlist("./load/test_networkxBeforeRemove.adjlist")
    else:
           return jsonify(err="405 Method Not Allowed")

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

    # networkx.draw(G, node_size=3)
    plt.savefig("./load/results/networkx_after_remove.png")
    app.logger.debug('finished load(G)')

    plt.close()
    return jsonify(before=before, after=after)
