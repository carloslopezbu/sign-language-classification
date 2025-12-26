import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import cv2

    return (YOLO,)


@app.cell
def _(YOLO):
    yolo = YOLO(model="yolo11n-pose.pt")
    return (yolo,)


@app.cell
def _(yolo):
    results = yolo("./datamining/CNSE/videos/USqNMZIZ21s.mp4", stream=True)
    return


app._unparsable_cell(
    r"""
    for res in results:
        img = res.orig_img
        kps = res.keypoints.xy[0].cpu().numpy()

        print(kps)

        for kp in kps:
            x, y = kp
            img = cv2.circle(img=img, center=(int(x),int(y)), radius=5, color=(255,0,0))

        mo.ui.
        plt.imshow(img)

    
    
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
