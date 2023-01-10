import plotly.express as px
import numpy as np
import pandas as pd

def to_plotly_test(feat_dict, truth_dict, file,additiona_dict={}, stringy=False, override_sids=None, range=None):
    print(truth_dict.keys())
    truth_assignment = truth_dict['truthHitAssignementIdx'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    result = {
        'x (cm)': feat_dict['recHitX'][:,0],
        'y (cm)': feat_dict['recHitY'][:,0],
        'z (cm)': feat_dict['recHitZ'][:,0],
        'rechit_energy': feat_dict['recHitEnergy'][:,0],
        't_only_minbias': truth_dict['t_only_minbias'][:,0],
        'truth_assignment': truth_assignment if override_sids is None else override_sids,
        'truth_assignment_energy': truth_dict['truthHitAssignedEnergies'][:,0],
        'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }

    for k,v in additiona_dict.items():
        result[k] = v[:, 0]

    result['t_only_minbias'] = np.array(['s'+str(x) for x in result['t_only_minbias']])

    result['size'] = np.log(result['rechit_energy']+1)
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)
    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="t_only_minbias", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template='plotly_dark' if file.endswith('.html') else 'ggplot2',
                        range_color=range,
                        color_continuous_scale=px.colors.sequential.Rainbow)

    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.write_html(file)

def to_plotly_true(feat_dict, truth_dict, file,  stringy=False, override_sids=None, range=None,
                   color_scale=px.colors.sequential.Rainbow, add_in_color=0, template=None, bigger_color_mapping=False, my_mapping=None):
    print(truth_dict.keys())
    truth_assignment = truth_dict['truthHitAssignementIdx'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    result = {
        'x (cm)': feat_dict['recHitX'][:,0],
        'y (cm)': feat_dict['recHitY'][:,0],
        'z (cm)': np.abs(feat_dict['recHitZ'][:,0]),
        'rechit_energy': feat_dict['recHitEnergy'][:,0],
        'truth_assignment': truth_assignment if override_sids is None else override_sids,
        'truth_assignment_energy': truth_dict['truthHitAssignedEnergies'][:,0],
        'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }

    if stringy:
        result['truth_assignment'] = ['s'+str(x) for x in (result['truth_assignment']+add_in_color)]

    result['size'] = np.log(result['rechit_energy']+1)
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)

    print("Uniques", np.unique(result['truth_assignment']))
    color_discrete_map = {
        's-1':'darkgray',
        's0':'blueviolet',
    }

    if bigger_color_mapping:
        color_discrete_map = {
            's-1': 'darkgray',
            's0': 'blueviolet',
            's1': 'crimson',
            's2': 'darkorange',
            's3': 'darkgreen',
            's4': 'darkturquoise',
            's5': 'lightpink',
            's6': 'red',
            's7': 'thistle',
            's8': 'cyan',
        }
    if my_mapping is not None:
        color_discrete_map = my_mapping.copy()


    if template is None:
        template = 'ggplot2' if file.endswith('.html') else 'ggplot2'

    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="truth_assignment", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template=template,
                        range_color=range,
                        # color_continuous_scale=color_scale,
                        color_discrete_map=color_discrete_map
                        )

    if file.endswith('.html'):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(file)
    else:
        tune_for_png(fig)
        fig.write_image(file,width=1500,height=1500)

def to_plotly_pred(feat_dict, pred_dict, file,  stringy=False, override_sids=None, range=None, color_scale=px.colors.sequential.Rainbow, add_in_color=0, template=None, bigger_color_mapping=False,
                   my_mapping=None):
    print(pred_dict.keys())
    pred_assignment = pred_dict['pred_sid'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    result = {
        'x (cm)': feat_dict['recHitX'][:,0],
        'y (cm)': feat_dict['recHitY'][:,0],
        'z (cm)': np.abs(feat_dict['recHitZ'][:,0]),
        'rechit_energy': feat_dict['recHitEnergy'][:,0],
        'pred_assignment': pred_assignment if override_sids is None else override_sids,
        'pred_assignment_energy': pred_dict['pred_energy'][:,0],
        # 'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }

    if stringy:
        result['pred_assignment'] = ['s'+str(x) for x in (result['pred_assignment']+add_in_color)]


    result['size'] = np.log(result['rechit_energy']+1)
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    color_discrete_map = {
        's-1': 'darkgray',
        's0': 'blueviolet',
    }

    if bigger_color_mapping:
        color_discrete_map = {
            's-1': 'darkgray',
            's0': 'blueviolet',
            's1': 'crimson',
            's2': 'darkorange',
            's3': 'darkgreen',
            's4': 'darkturquoise',
            's5': 'lightpink',
            's6': 'red',
            's7': 'thistle',
            's8': 'cyan',
        }

    if my_mapping is not None:
        color_discrete_map = my_mapping.copy()

    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)

    if template is None:
        template = 'ggplot2' if file.endswith('.html') else 'ggplot2'
    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="pred_assignment", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template=template,
                        range_color=range,
                        # color_continuous_scale=color_scale,
                        color_discrete_map=color_discrete_map
                        )

    fig.update_traces(marker=dict(line=dict(width=0)))

    if file.endswith('.html'):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(file)
    else:
        tune_for_png(fig)
        fig.write_image(file,width=1500,height=1500)

def to_plotly_beta(feat_dict, pred_dict, truth_dict, file,  stringy=False, override_sids=None):
    print(pred_dict.keys())
    # pred_assignment = pred_dict['pred_sid'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    result = {
        'a': pred_dict['pred_ccoords'][:,0],
        'b': pred_dict['pred_ccoords'][:,1],
        'c': pred_dict['pred_ccoords'][:,2],
        'beta': pred_dict['pred_beta'][:,0],
        'size': pred_dict['pred_beta'][:,0],
        'truth_assignment': truth_dict['truthHitAssignementIdx'][:,0] if override_sids is None else override_sids,
        # 'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }


    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)
    fig = px.scatter_3d(df, x="a", y="b", z="c",
                        color="truth_assignment", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template='plotly_dark',
                        color_continuous_scale=px.colors.sequential.Rainbow)

    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.write_html(file)

def tune_for_png(fig, camera=None):
    fontsize = 22
    fig.update_traces(marker=dict(line=dict(width=0)), showlegend=False, marker_showscale=False)

    # fig.update_xaxes(title_font={"size": fontsize})
    # fig.update_yaxes(title_font={"size": fontsize})
    # fig.update_zaxes(title_font={"size": fontsize})


    if camera is None:
        a = 1.25
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.1),
            eye=dict(x=-1.25 * a, y=1.15 * a, z=0.5 * a)
        )

    fig.update_layout(scene_camera=camera, font=dict(
        size=fontsize, ), scene=dict(xaxis=dict(tickfont=dict(size=18),
                                                title=dict(font=dict(size=25))),
                                     yaxis=dict(tickfont=dict(size=18),
                                                title=dict(font=dict(size=25))),
                                     zaxis=dict(tickfont=dict(size=18),
                                                title=dict(font=dict(size=25)))))

    fig.update_coloraxes(showscale=False)

def to_plotly(result_, file, plot_main=False, cut=0, compact=True, stringy=False, filter=None, plot_noise=False,
              plot_gev=-1, color_with=None, add=[], size_func=None, camera=None, lims_x=None, lims_y=None, lims_z=None):

    filt = np.greater(result_['rechit_energy'], cut)
    if filter is not None:
        filt = np.logical_and(filt, filter)

    truth_assignment = result_['truth_assignment'][filt]==0 if plot_main else result_['truth_assignment'][filt]
    truth_assignment = result_['truth_assignment'][filt]!=-1 if plot_noise else truth_assignment
    truth_assignment = np.where(result_['truth_assignment_energy'][filt]>plot_gev, truth_assignment, -1) if plot_gev!=-1 else truth_assignment
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    if stringy:
        truth_assignment = np.array(['s%d.'%x for x in truth_assignment.tolist()])

    result = {
        'x (cm)': result_['rechit_x'][filt],
        'y (cm)': result_['rechit_y'][filt],
        'z (cm)': result_['rechit_z'][filt],
        'rechit_energy': result_['rechit_energy'][filt],
        'truth_assignment': truth_assignment,
            'truth_assignment_energy': result_['truth_assignment_energy'][filt],
        'truth_assignment_energy_dep': result_['truth_assignment_energy_dep'][filt],
            # 'truth_assignment_pdgid': result_['truth_assignment_pdgid'][filt],
    }

    if not compact:
        result.update({
            'truth_assignment_hit_dep': result_['truth_assignment_hit_dep'][filt],
            'truth_assignment_vertex_position_x': result_['truth_assignment_vertex_position_x'][filt],
            'truth_assignment_vertex_position_y': result_['truth_assignment_vertex_position_y'][filt],
            'truth_assignment_vertex_position_z': result_['truth_assignment_vertex_position_z'][filt],
            'truth_assignment_momentum_direction_x': result_['truth_assignment_momentum_direction_x'][filt],
            'truth_assignment_momentum_direction_y': result_['truth_assignment_momentum_direction_y'][filt],
            'truth_assignment_momentum_direction_z': result_['truth_assignment_momentum_direction_z'][filt],
            'truth_assignment_energy_dep_all': result_['truth_assignment_energy_dep_all'][filt],
        })
    # print(result['truth_assignment'], len(np.unique(result['truth_assignment'])))

    if color_with is not None:
        result['color'] = result_[color_with][filt]
        if stringy:
            result['color'] = np.array(['s'+str(x) for x in result['color']])
    else:
        result['color'] = truth_assignment

    result['size'] = np.log(result['rechit_energy']+1) if size_func is None else size_func(result['rechit_energy'])
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    hover_data = [k for k,v in result.items()]
    for a in add:
        if a not in hover_data:
            result[a] = result_[a][filt]
            hover_data.append(a)
            # print("Adding", a)


    df = pd.DataFrame(result)
    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="color", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        # template='plotly_dark' if file.endswith('.html') else 'ggplot2',
                        template='ggplot2',
                        color_continuous_scale=px.colors.sequential.Rainbow)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=lims_z, ),
            yaxis=dict(range=lims_x, ),
            zaxis=dict(range=lims_y, ), ),)

    if file.endswith('.html'):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(file)
    else:
        tune_for_png(fig, camera=camera)
        fig.write_image(file,width=1500,height=1500)


