#!/usr/bin/env python

import PrinCoord
import sys, json, numpy, time, pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from pylab import *
#Uncomment the following line for use without a GUI:
#matplotlib.use("Agg")

class EmpData:
    """
    This is a class for processing the EMP Data
    """
    def __init__(self):
        """
          __init__(self)
            This is the instantiation of the object. 
        """

        self.biom = None
        self.mapping = None
        self.data = None

        #Keep the relevant arrays for created plots in memory as a dict
        #the keys are strings: feature+"-"+scale+"-"+alg
        #if scale is True, the values are: (feature_vals,colors)
        #if scale is False, the values are: (feature_strs,colors,recs,labels)
        self.plots = dict()

        #Keep the distance numpy arrays from the algorithms in memory as a dict
        self.fits = {"pcoa":None, "isomap":None}

        self.plot_args = { \
            "feature":None, "scale":None, "algorithm":None, "size":50, "colormap":"gist_rainbow", \
            "filter":None, "filter_scale":None, "include":None, "exclude":None, "min":None, "max":None}

        #These are all the defaults from a new sklearn.manifold.Isomap
        #except n_components, whose default is normally 2
        self.isomap_args = {"algorithm":"isomap", \
            "n_neighbors":5, "n_components":3, "eigen_solver":"auto", "tol":0, \
            "max_iter":None, "path_method":"auto", "neighbors_algorithm":"auto"}

        #The only option for PrinCoord is the distance measure
        self.pcoa_args = {"algorithm":"pcoa", \
            "distance":"hellinger"}

    #Import the biom file as a dict using json, then data is
    #the frequency matrix from the biom file, convert it to dense format,
    #and transpose it so the rows are samples and the columns are OTUs
    def load_biom(self, biomName):
        with open(biomName,"U") as file:
            self.biom = json.loads(file.read())
            self.data = self.dense(self.biom["data"],self.biom["shape"]).transpose()

    #Import the mapping file as a list of strings;
    #remove "#" from the first line and "\n" from every line
    def load_map(self, mapName):
        with open(mapName,"U") as file:
            self.mapping = file.readlines()
            self.mapping[0] = self.mapping[0].replace("#","")
            cleaned = []
            for line in self.mapping:
                cleaned.append(line.replace("\n",""))
            self.mapping = cleaned

    #If the requested fit is not in fits, calculate it and add it
    def fit(self, alg, opts=None):
        if(self.fits[alg] is None):
            start = time.time()
            if(alg=="pcoa"):
                opts = self.fill_args(opts,self.pcoa_args)
                self.fits["pcoa"] = PrinCoord.pcoa(self.data,distance=opts["distance"])[0:3].transpose()
            elif(alg=="isomap"):
                opts = self.fill_args(opts,self.isomap_args)
                self.fits["isomap"] = manifold.Isomap(n_neighbors=opts["n_neighbors"], \
                n_components=opts["n_components"],eigen_solver=opts["eigen_solver"], \
                tol=opts["tol"],max_iter=opts["max_iter"],path_method=opts["path_method"], \
                neighbors_algorithm=opts["neighbors_algorithm"]).fit_transform(self.data)
            print(alg+" took "+str(time.time()-start)+" seconds")

    #Calculate all available fits
    def fit_all(self):
        for key in self.fits:
            self.fit(key)

    #Delete a fit from the fits dict
    def clear_fit(self, alg):
        self.fits[alg] = None

    #Delete all fits from the dict
    def clear_all_fits(self):
        for key in self.fits:
            self.clear_fit(key)

    #Delete a plot from the plots dict
    def clear_plot(self, feature, scale, alg):
        key = feature+"-"+str(scale)+"-"+alg
        del self.plots[key]

    #Delete all plots from the plots dict
    def clear_all_plots(self):
        self.plots = dict()
    
    #Show the current plots
    def show(self):
        plt.show()

    #Clear the current plots and formatting
    def clear(self):
        plt.clf()

    #Save the current plots (useful if the system lacks a gui)
    def save(self, name):
        plt.savefig(name)

    #Pickle this EmpData object
    def pickle(self, name):
        with open(name, "w") as file:
            start = time.time()
            temp = self.data
            self.data = None
            pickle.dump(self,file,-1)
            self.data = temp
            print("pickling took "+str(time.time()-start)+" seconds")

    #Return the EmpData object from a pickle file
    #Just an easy way to unpickle in the command line interpreter
    def unpickle(self, name):
        with open(name, "r") as file:
            start = time.time()
            ed = pickle.load(file)
            ed.data = self.dense(ed.biom["data"],ed.biom["shape"]).transpose()
            print("unpickling took "+str(time.time()-start)+" seconds")
            return ed

    #Initialize defaults and replace with user specified
    def fill_args(self, opts, default):
        result = {}
        for key in default:
            result[key] = default[key]
        if opts is not None:
            for key in opts:
                result[key] = opts[key]
        return result

    #Create the requested 2d plots where shape=(rows,cols),
    #plots is an array of (feature,scale,alg), cmap is the colormap
    #to use if scale is True, and size is the size of the points
    def plots2d(self, shape, plots):
        (rows,cols) = shape
        for (i,p) in enumerate(plots):
            self.plot2d(self.fill_args(p,self.plot_args),100*rows+10*cols+i+1)

    #Create the requested 3d plots where shape=(rows,cols),
    #plots is an array of (feature,scale,alg), cmap is the colormap
    #to use if scale is True, and size is the size of the points
    def plots3d(self, shape, plots):
        (rows,cols) = shape
        fig = plt.figure()
        for (i,p) in enumerate(plots):
            self.plot3d(fig,self.fill_args(p,self.plot_args),100*rows+10*cols+i+1)

    #Create the arrays for the requested plots and add them to the dict
    #the keys are strings: feature+"-"+scale+"-"+alg
    #if scale is True, the values are: feature_vals
    #if scale is False, the values are: (feature_strs,colors,recs,labels)
    def add_plot(self, plot):
        feature=plot["feature"]; scale=plot["scale"]; alg=plot["algorithm"];
        key = feature+"-"+str(scale)+"-"+alg
        if key not in self.plots:
            if scale:
                value = self.feature_vals(feature)
            else:
                value = self.assign_colors(self.feature_strs(feature))
            self.plots[key] = value

    #Add the requested 2d plot
    def plot2d(self, plot, sub):
        feature=plot["feature"]; scale=plot["scale"]; alg=plot["algorithm"];
        self.fit(alg)
        start = time.time()
        dist = self.fits[alg]
        plt.subplot(sub)
        key = feature+"-"+str(scale)+"-"+alg
        self.add_plot(plot)
        if(scale):
            vals = self.plots[key]
            if plot["filter"] is not None:
                (dist,vals) = self.filter_num(dist,vals,plot)
            sc = plt.scatter(dist[:,0],dist[:,1],c=vals,cmap=plot["colormap"],s=plot["size"])
            plt.colorbar(sc)
        else:
            (strs, colors, recs, labels) = self.plots[key]
            if plot["filter"] is not None:
                (dist,strs,colors,recs,labels) = self.filter_cat(dist,strs,colors,recs,labels,plot)
            plt.scatter(dist[:,0],dist[:,1],c=colors,s=plot["size"])
            legend(recs, labels)
        plt.title(feature+" ("+alg+")")
        print("plot "+key+" took "+str(time.time()-start)+" seconds")

    #Add the requested 3d plot
    def plot3d(self, fig, plot, sub):
        feature=plot["feature"]; scale=plot["scale"]; alg=plot["algorithm"];
        self.fit(alg)
        start = time.time()
        dist = self.fits[alg]
        ax = fig.add_subplot(sub, projection="3d")
        key = feature+"-"+str(scale)+"-"+alg
        self.add_plot(plot)
        if(scale):
            vals = self.plots[key]
            if plot["filter"] is not None:
                (dist,vals) = self.filter_num(dist,vals,plot)
            sc = ax.scatter(dist[:,0],dist[:,1],dist[:,2],c=vals,cmap=plot["colormap"],s=plot["size"])
            fig.colorbar(sc);
        else:
            (strs, colors, recs, labels) = self.plots[key]
            if plot["filter"] is not None:
                (dist,strs,colors,recs,labels) = self.filter_cat(dist,strs,colors,recs,labels,plot)
            ax.scatter(dist[:,0],dist[:,1],dist[:,2],c=colors,s=plot["size"])
            legend(recs,labels)
        ax.set_title(feature+" ("+alg+")")
        print("plot "+key+" took "+str(time.time()-start)+" seconds")

    #Return the filtered distance matrix and values
    def filter_num(self, dist, vals, plot):
        fdist = []; filvals = []
        if plot["filter_scale"]:
            fmin = plot["min"]; fmax = plot["max"]
            cvals = self.feature_vals(plot["filter"])
            for i in range(0,len(dist)):
                if (fmin is None and cvals[i]<=fmax) \
                or (fmax is None and cvals[i]>=fmin) \
                or (cvals[i]>=fmin and cvals[i]<=fmax):
                    fdist.append(dist[i])
                    filvals.append(vals[i])
        else:
            cstrs = self.feature_strs(plot["filter"])
            for i in range(0,len(dist)):
                if (plot["exclude"] is None and cstrs[i] in plot["include"]) \
                or (plot["include"] is None and cstrs[i] not in plot["exclude"]):
                    fdist.append(dist[i])
                    filvals.append(vals[i])
        return (numpy.array(fdist),filvals)

    #Return the filtered distance matrix and values
    def filter_cat(self, dist, strs, colors, recs, labels, plot):
        fdist = []; fstrs = []; fcolors = []; frecs = []; flabels = []
        if plot["filter_scale"]:
            fmin = plot["min"]; fmax = plot["max"]
            cvals = self.feature_vals(plot["filter"])
            for i in range(0,len(dist)):
                if (fmin is None and cvals[i]<=fmax) \
                or (fmax is None and cvals[i]>=fmin) \
                or (cvals[i]>=fmin and cvals[i]<=fmax):
                    fdist.append(dist[i])
                    fstrs.append(strs[i])
                    fcolors.append(colors[i])
        else:
            cstrs = self.feature_strs(plot["filter"])
            for i in range(0,len(dist)):
                if (plot["exclude"] is None and cstrs[i] in plot["include"]) \
                or (plot["include"] is None and cstrs[i] not in plot["exclude"]):
                    fdist.append(dist[i])
                    fstrs.append(strs[i])
                    fcolors.append(colors[i])
        (cDict,frecs,flabels) = self.make_legend(fstrs,fcolors)
        return (numpy.array(fdist),fstrs,fcolors,frecs,flabels)

    #Return the values for the feature in the mapping file as strings
    def feature_strs(self, feature):
        features = self.mapping[0].split("\t")
        fCol = features.index(feature)
        strs = []
        for row in range(0,len(self.data)):
            sample = self.biom["columns"][row]["id"]
            mapRow = [r for (meta,r) in zip(self.mapping,range(0,len(self.mapping))) if sample in meta][0]
            s = self.mapping[mapRow].split("\t")[fCol]
            strs.append(s)
        return strs

    #Return the values for the feature in the mapping file as floats
    def feature_vals(self, feature):
        strs = self.feature_strs(feature)
        vals = [float(s) for s in strs]
        return vals

    #Return an array of colors for the categorical feature
    def assign_colors(self, strs):
        (cDict, recs, labels) = self.make_legend(strs, None)
        colors = []
        for s in strs:
            colors.append(cDict[s])
        return (strs, colors, recs, labels)

    #Return the Rectangles and labels for a new legend
    def make_legend(self, strs, colors):
        unique = set(strs)
        cDict = dict()
        if colors is None:
            allColors = ["b","g","r","c","m","y","k","w"]
        else:
            allColors = set(colors)
        recs = []
        labels = []
        for (i,s,c) in zip(range(0,len(unique)),unique,allColors):
            cDict[s] = c
            recs.append(mpatches.Rectangle((0,0),1,1,fc=cDict[s]))
            labels.append(s)
        return (cDict, recs, labels)

    #Return a dense representation of the data numpy array
    def dense(self, data, shape):
        d = numpy.zeros(shape)
        for point in data:
            d[point[0],point[1]] = point[2]
        return d

    #Return a sparse representation of the data numpy array
    def sparse(self, data):
        d = []
        for (r,row) in enumerate(data):
            for (c,col) in enumerate(row):
                val = data[r,c]
                if val != 0:
                    d.append([r,c,val])
        return numpy.array(d)
