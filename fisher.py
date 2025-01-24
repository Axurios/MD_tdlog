from dataHolder import DataHolder
import numpy as np

def fisher_theta(Data : DataHolder, gradstr : str, forcestr: str, beta : float):

    keys  = list(Data.md_data.keys())   # getting each experience
    G_list=[]                           # list of gradient of descriptor
    F_list = []                         # list of Forces
    GGT = []

    for key in keys :
        for atoms in Data.md_data[key]['atoms']:

            # Getting data in the lists defined above, and reshaping them
            grad_mat = np.array(atoms.get_array(gradstr))
            sh = list(grad_mat.shape)
            F_list.append(np.array(atoms.get_array(forcestr)).reshape(-1))
            G_list.append(grad_mat.reshape(-1,sh[-1]))


    #computing theta based on Fisher
    for G in G_list :
        GGT.append(np.dot(G.transpose(),G))
    c_list = np.array([(beta**2)*np.dot(G_list[i].transpose(),F_list[i]) for i in range(len(G_list))])
    stacked_c = np.stack(c_list)
    c = np.mean(stacked_c, axis = 0)
    stacked_arrays = np.stack(GGT)
    T = np.mean(stacked_arrays, axis=0)
    T = beta**2 * T

    theta_dot = np.linalg.solve(T, c)

    return theta_dot