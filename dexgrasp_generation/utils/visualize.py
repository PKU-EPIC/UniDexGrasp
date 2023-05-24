import torch
import plotly.graph_objects as go

def visualize(hand_model, object_pc, hand_pose_pred, hand_pose_gt=None, show_num=1e10):
    show_num = min(show_num, hand_pose_pred.shape[0])
    pc = object_pc.detach().cpu().numpy()
    object_pc_plotly = go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], 
            mode='markers', marker=dict(
                size=2, 
                colorbar=dict(title="signed squared distance"), 
                colorscale='Jet'
            )
        )
    fig_list = [object_pc_plotly]

    if hand_pose_gt is not None:
        hand = hand_model(hand_pose_gt, object_pc[None, :, :], with_surface_points=True)
        v = hand['surface_points'][0].detach().cpu().numpy()
        hand_plotly = go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(color='lightpink', size=2))
        fig_list.append(hand_plotly)

    for i in range(show_num):
        hand_pose_pred_ = hand_pose_pred[i:i+1]
        hand_pred = hand_model(hand_pose_pred_, object_pc[None, :, :], with_meshes=True)

        v_pred = hand_pred['vertices'][0].detach().cpu().numpy()
        f_pred = hand_pred['faces'].detach().cpu().numpy()
        hand_pred_plotly = go.Mesh3d(x=v_pred[:, 0], y=v_pred[:, 1], z=v_pred[:, 2], i=f_pred[:, 0], j=f_pred[:, 1], k=f_pred[:, 2], color='lightblue', opacity=1)
        
        fig_list.append(hand_pred_plotly)

    fig = go.Figure(fig_list)
    fig.show()
    print("visualize")