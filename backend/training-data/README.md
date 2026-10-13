# EffectLab Scratch training data

Place one subfolder per aligned training sequence in this directory. Do not commit private footage or very large videos to GitHub; use a local external directory and point DATA_DIR at it instead.

Each sequence folder must contain:

- reference.jpg: a full-body reference photo of the identity the model should render.
- driving.mp4: the source performer, camera and background whose motion should be followed.
- target.mp4: the intended replacement identity performing the same movement, with the same camera timing and frame alignment as driving.mp4.

Example:

~~~text
training-data/
  person_a_dance_01/
    reference.jpg
    driving.mp4
    target.mp4
  person_b_walk_02/
    reference.jpg
    driving.mp4
    target.mp4
~~~

Training pairs need to be genuine target labels. Do not copy driving.mp4 to target.mp4. The audit tool detects byte-identical files and reports mismatched frame rates, frame counts or duration. It also writes contact sheets for manual visual review.

For reliable identity replacement, use many identities, clothing styles, full-body views, body shapes, actions, camera movements and lighting conditions. A handful of clips can only overfit; it will not create a general-purpose video AI. Each driving/target pair should be captured or created so motion and framing closely match. Get permission from everyone shown and for all footage.

Run the data audit from the backend folder:

~~~bash
python prepare_data.py --data training-data --report training-audit
~~~

The audit verifies basic timing; it cannot automatically guarantee semantic alignment. Open every contact sheet and inspect the target and driving images at each timepoint before training.
