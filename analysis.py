#### IMPORTS ####

import argparse as arg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts, distances
import mdtraj as mdt
from scipy.ndimage import gaussian_filter
import os
import time
import umap
import hdbscan
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#### NOTES ####

# Contact analysis DONE
# Surface free energy DONE
# PCA to identify large scale motions DONE
# H bond analysis

# To dos
# Push code to github
# Clean up functions. Start with changing naming convention
# 1. Perhaps changing data["rbd"] to the actual description "protein and chainID A and not name H*"
# 2. Clean up code. Ex, instances of where new variable is defined for data[key]. Just use data[key]
# 3. load in mdtraj in makeU. Perhas call that the setup function. No other file should require files

# Push to github again
# Test script on all the nb designed

# Long term
# Clean repository. Folders
# 1. Nb generation
# 2. code used to run simulations
# 3. analysis
# Write a graphing function, make if statement in functions if visualization == True: ...

#### FUNCTIONS ####

def setup():
    data = {}
    return data

def makeU(files, data):
    print("\nMaking universe..")
    data["u"] = mda.Universe(files["pdb"], files["dcd"])
    data["rbd"] = data["u"].select_atoms("protein and chainID A and not name H*")
    data["nb"] = data["u"].select_atoms("protein and chainID B and not name H*")
    return data

# https://userguide.mdanalysis.org/1.0.0/examples/analysis/distances_and_contacts/contacts_native_fraction.html
def findContacts(data, verbose = True):
    print("\nFinding contact frequency over trajectory..")
    rbdSel, nbSel = "protein and chainID A and not name H*", "protein and chainID B and not name H*"
    # start = time.perf_counter()
    c = contacts.Contacts(data["u"],
                          select = (rbdSel, nbSel),
                          refgroup = (data["rbd"], data["nb"]),
                          radius = data["cutoff"],
                          method = "radius_cut")
    c.run(verbose = verbose)
    # end = time.perf_counter()
    # print(f"Contacts found in {end - start} seconds.")
    data["contacts"] = pd.DataFrame(c.results.timeseries,
                                    columns = ["Frame", "Contacts"])
    
    # Visualization
    data["contacts"].plot(x = "Frame")
    plt.ylabel("Fraction of contacts")
    plt.savefig(f"Contacts{data['cutoff']}A.png")
    plt.close()
    return data

# https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_between_selections.html
# https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_within_selection.html
def findDist(data, verbose = True):
    print("\nFinding average pairwise distances..")
    rbdCom= data["rbd"].center_of_mass(compound = "residues")
    nbCom = data["nb"].center_of_mass(compound = "residues")
        
    dL = [ ]
    start = time.perf_counter()
    for step in data["u"].trajectory:        
        d = distances.distance_array(nbCom, # reference (y)
                                     rbdCom, # configuration (x)
                                     box= data["u"].dimensions) # Angstroms = 0.1*Nanometers
        # need to learn how to make a python progress bar            
        dL.append(d)
    
    end = time.perf_counter()
    print(f"Finished calculating distances in {end - start} seconds")
    dA = np.array(dL)
    data["distance"] = np.mean(dA, axis=0) # average of d over frames. axis = 0 is average over depth/layers (frames)
    
    # Visualization
    fig0, ax0 = plt.subplots()
    im0 = ax0.imshow(data["distance"], origin='upper')
    tick_interval = 5
    ax0.set_xticks(np.arange(len(rbdCom))[::tick_interval])
    ax0.set_yticks(np.arange(len(nbCom))[::tick_interval])
    # ax2.set_xticklabels(data["rbd"].residues.resids[::tick_interval])
    # ax2.set_yticklabels(data["nb"].residues.resids[::tick_interval])\
    plt.xlabel('RBD')
    plt.ylabel('NB')
    plt.title('Average Distance Between Nb and Rbd Residues')
    plt.xticks(rotation=90)
    cbar0 = fig0.colorbar(im0)
    cbar0.ax.set_ylabel('Distance (Å)')
    plt.savefig("ResidueDistanceHeatmap.png")
    plt.close()
    
    # Filtering and more plotting
    mask = data["distance"] <= data["cutoff"]
    data["filteredDistance"] = np.where(mask, data["distance"], 0)
    
    fig1, ax1 = plt.subplots(figsize=(8,6))
    im1 = ax1.imshow(data["filteredDistance"], origin='upper', cmap='viridis', interpolation='none')
    tick_interval = 5
    ax1.set_xticks(np.arange(len(rbdCom))[::tick_interval])
    ax1.set_yticks(np.arange(len(nbCom))[::tick_interval])
    plt.xlabel('RBD')
    plt.ylabel('NB')
    plt.title(f'Filtered Residue Interactions (Cutoff = {data["cutoff"]})')
    plt.xticks(rotation=90)
    # Colorbar
    cbar1 = fig1.colorbar(im1)
    cbar1.ax.set_ylabel('Distance (Å)')
    plt.tight_layout()
    plt.savefig("ResidueDistanceFilteredHeatmap.png")
    plt.close()
    
    return data

def findContacts2(data, verbose = True):
    print("\nFinding pairwise contacts")
    # heat map needs ticklabels indicates residue number, name, and atom name

    # plt.figure(figsize=(len(xlabels)*.2+1,len(ylabels)*.2+1))
    # sns.heatmap(contacts,xticklabels=xlabels,yticklabels=ylabels, vmin=0,vmax=1, cmap="viridis",cbar_kws={'pad':0.01,'label':'Contact Frequency'})
    # plt.tight_layout()
    # plt.savefig(f'{args.output_prefix}_contacts.png',bbox_inches='tight',dpi=300)
    rbdCom= data["rbd"].center_of_mass(compound = "residues")
    nbCom = data["nb"].center_of_mass(compound = "residues")
        
    contactA = np.zeros((len(nbCom), len(rbdCom))) # initiailized contact matrix for each pairwise contact. Adds 1 each time distance is less than cutoff 
    start = time.perf_counter()
    for step in data["u"].trajectory:        
        d = distances.distance_array(nbCom, # reference (y)
                                     rbdCom, # configuration (x)
                                     box= data["u"].dimensions) # Angstroms = 0.1*Nanometers
        contactA += (d < data["cutoff"]) # add contact if less than cutoff
    
    end = time.perf_counter()
    print(f"Finished finding contacts in {end - start} seconds")
    
    contactA = contactA / len(data["u"].trajectory) # Frequency
    mask = contactA > 0.5 # Filtering out contact times less than 50%
    contactAfiltered = np.where(mask, contactA, 0)
    
    fig0, ax0 = plt.subplots()
    im0 = ax0.imshow(contactAfiltered, origin='upper')
    tick_interval = 5
    ax0.set_xticks(np.arange(len(rbdCom))[::tick_interval])
    ax0.set_yticks(np.arange(len(nbCom))[::tick_interval])
    # ax2.set_xticklabels(data["rbd"].residues.resids[::tick_interval])
    # ax2.set_yticklabels(data["nb"].residues.resids[::tick_interval])\
    plt.xlabel('RBD')
    plt.ylabel('NB')
    plt.title('Contact Frequency Between Nb and Rbd Residues')
    plt.xticks(rotation=90)
    cbar0 = fig0.colorbar(im0)
    cbar0.ax.set_ylabel('Frequency')
    plt.savefig("ResidueContactHeatmap.png")
    plt.close()
    
    # Filtering and more plotting
    # mask = data["distance"] <= data["cutoff"]
    # data["filteredDistance"] = np.where(mask, data["distance"], 0)
    
    # fig1, ax1 = plt.subplots(figsize=(8,6))
    # im1 = ax1.imshow(data["filteredDistance"], origin='upper', cmap='viridis', interpolation='none')
    # tick_interval = 5
    # ax1.set_xticks(np.arange(len(rbdCom))[::tick_interval])
    # ax1.set_yticks(np.arange(len(nbCom))[::tick_interval])
    # plt.xlabel('RBD')
    # plt.ylabel('NB')
    # plt.title(f'Filtered Residue Interactions (Cutoff = {data["cutoff"]})')
    # plt.xticks(rotation=90)
    # # Colorbar
    # cbar1 = fig1.colorbar(im1)
    # cbar1.ax.set_ylabel('Distance (Å)')
    # plt.tight_layout()
    # plt.savefig("ResidueDistanceFilteredHeatmap.png")
    # plt.close()
    return data

def calcPca(files, data, verbose = True):
    t = mdt.load(files["dcd"], top = files["pdb"])
    t.superpose(t, 0) # align with 0th frame
    
    atoms = t.topology.select("name CA")
    coords = t.xyz[:, atoms, :].reshape(len(t), -1) # flatten to 2d matrix for PCA
    # flattened rows = frames: [x1 y1 z1 x2 y2 z2 ... xn yn zn]
    # columns = features
    
    nFeatures = coords.shape[1]
    coordsCentered = coords - coords.mean(axis=0) # centered matrix for PCA
    cov = (coordsCentered.T @ coordsCentered) / (nFeatures - 1)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    data["pc"] = coordsCentered@eigvecs # [frames x features] @ [features @ features]
    return data
    
def calcSfe(data, bins = 200, sigma = 2, temperature = 310):
    print("Calculating surface free energy")
    kB = 1.380649E-23 # J/K
    T = 273.15 + temperature # K
    
    states, xedges, yedges = np.histogram2d(data["pc"][:,0], 
                                       data["pc"][:,1], 
                                       bins=bins)
    prob = states / np.sum(states) # probability of a state (bin)
    F = -kB*T*np.log(prob, where=(prob>0)) # absolute free energy
    deltaF = F - np.nanmin(F) # relative free energy, sets min to 0

    data["deltaFsmooth"] = gaussian_filter(deltaF, sigma = sigma)
    
    # Visualization
    plt.contourf(data["deltaFsmooth"].T, cmap='viridis')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="ΔF (kcal/mol)")
    plt.savefig("sfePCA.png")
    return data

# https://pubs.acs.org/doi/10.1021/acs.jcim.5c01725#fig1
def findResCor(data):
    data["rbd"] = data["u"].select_atoms("chainID A and  name CA")
    data["nb"] = data["u"].select_atoms("chainID B and name CA")

    nbN = len(data["nb"])
    rbdN = len(data["rbd"])
    nFrames = len(data["u"].trajectory)

    # Initialize empty arrays for position (x, y ,z) averages of residues
    nbAvgPos = np.zeros((nbN, 3))
    rbdAvgPos = np.zeros((rbdN, 3))

    for step in data["u"].trajectory:
        nbAvgPos += data["nb"].positions
        rbdAvgPos += data["rbd"].positions

    nbAvgPos /= nFrames
    rbdAvgPos /= nFrames

    # Compute inter-protein covariance
    # C(i,j) = < ΔR_i • ΔR_j >
    cov = np.zeros((nbN, rbdN))
    varNB  = np.zeros(nbN)
    varRBD = np.zeros(rbdN)

    # Centering the matrix
    for step in data["u"].trajectory:
        nbDelta = data["nb"].positions - nbAvgPos  # nb x 3
        rbdDelta = data["rbd"].positions - rbdAvgPos  # rbN x 3
        # Compute dot product for each pair (i,j)
        cov += nbDelta @ rbdDelta.T  # nb x rbN
        varNB  += np.sum(nbDelta**2,  axis=1)
        varRBD += np.sum(rbdDelta**2, axis=1)

    cov /= nFrames  # ensemble average
    varNB  /= nFrames
    varRBD /= nFrames

    # Compute normalized correlation
    # C(i,j) = c(i,j) / sqrt(c(i,i) * c(j,j))

    norm = np.zeros_like(cov)
    for i in range(nbN):
        for j in range(rbdN):
            denom = np.sqrt(varNB[i] * varRBD[j])
            norm[i, j] = cov[i, j] / denom if denom != 0 else 0


    # Apply distance cutoff for close contacts
    # Average distances between residues
    avgD = np.zeros((nbN, rbdN))
    for i in range(nbN):
        for j in range(rbdN):
            avgD[i, j] = np.linalg.norm(nbAvgPos[i] - rbdAvgPos[j])

    # Distance cutoff: 8 Å for positive correlations, 13 Å for negative correlations
    cutoffM = np.zeros_like(norm)
    cutoffM[(norm > 0) & (avgD <= 8)] = norm[(norm > 0) & (avgD <= 8)]
    cutoffM[(norm < 0) & (avgD <= 13)] = norm[(norm < 0) & (avgD <= 13)]
    
    data["attractionScore"] = np.sum(cutoffM[cutoffM > 0])
    data["repulsionScore"]  = np.sum(cutoffM[cutoffM < 0])
    data["bindingScore"]    = np.sum(cutoffM)
    
    # Visualization
    im = plt.imshow(cutoffM, cmap="coolwarm", vmin=-1, vmax=1, origin='upper')
    tick_interval = 5
    plt.xticks(np.arange(0, rbdN, tick_interval))
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, nbN, tick_interval))
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation")
    plt.xlabel("RBD")
    plt.ylabel("NB")
    plt.title("Close-Contact Inter-Protein Covariance Matrix")
    plt.savefig("ResidueCorrelation.png", dpi=300, bbox_inches='tight')
    plt.close()
    return data

def runUmap(data, n_neighbors=90, min_dist=0.6, n_components=3):
    print("Running UMAP..")
    start = time.perf_counter()
    pcs = data["pc"][:, :10]   # first 10 PCs
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=42
    )
    
    emb = reducer.fit_transform(pcs)
    data["umap"] = emb
    
    end = time.perf_counter()
    print(f"Finished finding contacts in {end - start} seconds")
    
    # Graphing
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=np.arange(len(emb)), cmap="viridis", s=3)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Frame index")
    plt.title("UMAP of PCA Projection")
    plt.savefig(f"umapN{n_neighbors}D{min_dist}.png")
    plt.close()
    
    return data
    
#     return data
def clusterUmap(data, min_cluster_size=500):
    print("Clustering UMAP with no noise..")
    emb = data["umap"]

    # Step 1: HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=0.01
    )
    labels = clusterer.fit_predict(emb)

    # Step 2: Assign noise points (-1) to nearest cluster
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        unique_labels = np.unique(labels[labels != -1])
        centroids = np.array([emb[labels == lbl].mean(axis=0) for lbl in unique_labels])

        for i in noise_idx:
            nearest = np.argmin(cdist([emb[i]], centroids))
            labels[i] = unique_labels[nearest]

    data["cluster"] = labels

    # Step 3: Plot with legend instead of colorbar
    plt.figure(figsize=(8,6))
    unique_clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for idx, cluster_id in enumerate(unique_clusters):
        cluster_points = emb[labels == cluster_id]
        plt.scatter(
            cluster_points[:,0], 
            cluster_points[:,1], 
            color=cmap(idx % 10), 
            s=4, 
            label=f"Cluster {cluster_id}"
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Clustered UMAP (HDBSCAN, no noise)")
    plt.legend(markerscale=3, fontsize=10)
    plt.savefig("clustered_umap.png", dpi=300)
    plt.close()

    return data



### RUN ####

def runAll(pdb = "system.pdb", 
           dcd = "trajectory.dcd"):
    data = {"cutoff": 0.4}
    files = {
        "pdb": pdb,
        "dcd": dcd
    }
    
    data = makeU(files, data)
    # data = findDist(data)
    # data = findResCor(data) GOOD
    # data = findContacts2(data) # contact frequency between residues GOOD
    # data = findContacts(data) # contact frequency change over simulation GOODish
    data = calcPca(files, data)
    # data = calcSfe(data) # GOOD
    
    # data = analyze_binding_modes(files, data)
    data = runUmap(data)
    data = clusterUmap(data)
    # data = calcRmsd(data)
    
    return None

if __name__ == "__main__":
    cwDir = os.getcwd()
    filesDir = "/nbD1Run24Nov25"
    os.chdir(cwDir + filesDir)
    runAll()
    
    
#### NOTES ####

# Fast distances caluclations: https://docs.mdanalysis.org/stable/documentation_pages/lib/distances.html#MDAnalysis.lib.distances.distance_array


# Original umap function including noise
# def clusterUmap(data, min_cluster_size=500):
#     print("Clustering UMAP..")
#     emb = data["umap"]
    
#     # HDBSCAN = best for UMAP landscapes
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=1,
#         cluster_selection_epsilon=0.01
#     )
    
#     labels = clusterer.fit_predict(emb)
#     data["cluster"] = labels

#     # Plot
#     plt.figure(figsize=(8,6))
#     plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="tab10", s=4)
#     plt.xlabel("UMAP1")
#     plt.ylabel("UMAP2")
#     plt.title("Clustered UMAP (HDBSCAN)")
#     plt.colorbar(label="Cluster ID")
#     plt.savefig("clustered_umap.png", dpi=300)
#     plt.close()

# def analyze_binding_modes(files, data):
#     # ---- Load + Align ----
#     traj = mdt.load(files["dcd"], top=files["pdb"])
#     traj.superpose(traj, 0)

#     # ---- PCA ----
#     atoms = traj.topology.select("name CA")
#     coords = traj.xyz[:, atoms, :].reshape(len(traj), -1)
#     coords_centered = coords - coords.mean(axis=0)

#     cov = coords_centered.T @ coords_centered / (coords_centered.shape[1] - 1)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     idx = np.argsort(eigvals)[::-1]
#     eigvecs = eigvecs[:, idx]

#     pcs = coords_centered @ eigvecs
#     pc1, pc2 = pcs[:, 0], pcs[:, 1]

#     # ---- KMeans: identify PCA basins ----
#     kmeans = KMeans(n_clusters=3, random_state=0)
#     labels = kmeans.fit_predict(pcs[:, :2])

#     data["pc"] = pcs
#     data["cluster"] = labels

#     # ---- Plot PCA clusters ----
#     plt.figure()
#     plt.scatter(pc1, pc2, c=labels, cmap="viridis", s=2)
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.colorbar(label="Cluster")
#     plt.savefig("PCA_clusters.png")
#     plt.close()

#     # ---- Contact calculation setup ----
#     u = data["u"]
#     rbd_com = data["rbd"].center_of_mass(compound="residues")
#     nb_com = data["nb"].center_of_mass(compound="residues")

#     n_nb = len(nb_com)
#     n_rbd = len(rbd_com)
#     cutoff = data["cutoff"]

#     # ---- One contact map per cluster ----
#     contact_maps = {c: np.zeros((n_nb, n_rbd)) for c in range(3)}
#     counts = {c: 0 for c in range(3)}

#     for i, ts in enumerate(u.trajectory):
#         c = labels[i]  # cluster of this frame
        
#         d = distances.distance_array(nb_com, rbd_com, box=u.dimensions)
#         contact_maps[c] += (d < cutoff).astype(float)
#         counts[c] += 1

#     # ---- Normalize + Plot ----
#     for c in range(3):
#         cm = contact_maps[c] / counts[c]

#         plt.figure(figsize=(8, 6))
#         plt.imshow(cm, origin='upper', cmap='viridis', vmin=0, vmax=1)
#         plt.title(f"Cluster {c} Contact Map")
#         plt.xlabel("RBD residues")
#         plt.ylabel("NB residues")
#         plt.colorbar(label="Contact Frequency")
#         plt.savefig(f"cluster_{c}_contacts.png")
#         plt.close()

#     return data



# UMAP 90 and 0.6 worked for NB1