import math
import matplot.pyplot as plt
import sys

def plot_multiple(plot_function, args, filename=None, figsize=(14,9), label_left = "Density", 
                  label_bottom="Coarse Grain Measure Value", label_right="Energy", 
                  legend_pos=(1.40,0.90), legend_labels=None):
    
    fig = plt.figure(figsize=figsize)
    #figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    num_structs = len(args)
    rows = int(1.5 * math.sqrt(num_structs))
    cols = int(math.ceil(num_structs / float(rows)))
    ls,lb = None,None
    
    #figsize(cols * 4, rows * 3)
    
    for i, arg in enumerate(args):
        ax = fig.add_subplot(rows, cols, i+1)
        try:
            (ls, lb) = plot_function(arg, ax=ax)
        except Exception:
            import traceback
            print >>sys.stderr, traceback.format_exc()
            #print >>sys.stderr, "Error:", str(e)
            continue
            
    fig.add_subplot(rows, cols, 1)
    ax = fig.add_subplot(rows, cols, num_structs)
    #ls1, lb1 = ax1.get_legend_handles_labels()
    
    if label_bottom is not None:
        fig.text(0.5, 0.00, label_bottom, ha='center', va='center', fontsize=13)
        
    if label_left is not None:
        fig.text(0.00, 0.60, label_left, ha='center', va='center', rotation='vertical', fontsize=13)
    
    if label_right is not None:
        fig.text(1., 0.60, label_right, ha='center', va='center', rotation='vertical', fontsize=13)
    
    plt.tight_layout()
    plt.subplots_adjust(top=1.30)
    
    if legend_labels is not None:
        lb = legend_labels
    
    if ls is not None and lb is not None:
        plt.legend(ls, lb, bbox_to_anchor=legend_pos, loc=2, borderaxespad=0.)
    
    if filename is not None:
        # blah
        plt.savefig(filename, bbox_inches='tight')
        
    return ax
