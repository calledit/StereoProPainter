# StereProPainter

Fasten your seat belt Dorothy, cause Stereo Disocclusion is going bye-bye.


## Problem space
Inpainting video frames can be broken down in to two parts.
1. Taking stuff that that has been seen in earlier or later frames and placing them in the current frame.
2. Inpainting stuff that you don’t have any information about. IE guessing about how the scene looks.

**Part 1.** Is probably best done by some type of splating with a world model. This would however require a very very powerful model and computer especially since it needs to deal with moving objects in the scene. You can pick a simpler but less reliable way to do it. Where you only look at recent frames and in paint small areas, which is how propainter deal with this part.

**Part 2.** To make the theoretically best guesses you would have to interrogate the world model from part 1, that way you could guess based on all available info avalible in the video, both forward and backward in time. They way propainter deals with this is a simple gated infill model. A model that relies on L1 loss and a GAN loss. When you are guessing on infill. You have two problems you have realism and you have consistency. To get the best consistency you have to rely 100% on L1 loss, from that you get a blurry infill which has the most likely colors for each pixel. This looks terrible but it is the absolute optimum way of solving the consistency as good as you can. To get realism you use GAN Loss. With GAN loss you get actual guesses for what was in the missing area. But those guesses will be wrong to some extent since it does not have the info it needs to make the correct guess 100% of the time. Each guess the GAN make will be wrong in a different way and that means consistency is no more.

So you have two axises realism and consistency. Where High realism = Low consistency and High consistency = Low realism.
The way to optimise this is to find somewhere on that line that is as consistent enough and as realistic as possible.

What we want is to find the optimum point on the line. The optimum point being as far up in to the realism direction as posible without the image looking perceptually inconsistent.

<img width="500" height="675" alt="Screenshot 2026-03-28 at 00 16 51" src="https://github.com/user-attachments/assets/326baecf-6cb1-4cfd-8de8-cb2a05989d13" />


In propainter this is done by selecting the adversarial_weight by default propainter has this set to 0.01


## What was changed

To make StereProPainter better at stereo inpainting the adversarial_weight was increased as the holes are quite small and making correct guesses should be fairly easy. A specific issue with stereo inpainting is that you often have to draw over the edge between foreground and background and this does not look good when blurred this mean we have a bigger incentive than in many other infill scenarios to push up the realism as it means we get rid of the blur that is extra problematic in stereo inpainting.

While most holes are small there are still is sections where the holes are quite big in these holes you dont want the model to guess about details in the image. You want to rely more on L1 loss there. If you let the descriminator make the generator guess about what should be in those areas it will halucinate wildly. In test anything that is more than about 10 pixels away from real known data and has complex pattern (ie the image is not obvoiusly something like a solid colored wall) will start to be infilled with halucinated artifacts.

StereProPainter way to deal with this can be to blur the distant regions when traning the descriminator. That works but since bluring with a simple kernel and bluring as produced with an ai tranied with L1 loss are slightly diffrent that may lead to worse results. The best solution would be to blur using a network that was only trained on a L1 loss. But that is in practicallity to expensive.
On top of that a second disriminator was added.

Since there is a certain patern to how stereo infill masks are created there is also a certain patern to how one can infill them to make the model better att using that pattern in its inpanting StereProPainter was finetuned with reprojected images.

# Result

https://github.com/user-attachments/assets/2aba52fc-c5f5-427c-8048-628167006e5e

# Right eye video

### Low fps videos:
Right eye infilled video of cat:
* https://github.com/calledit/StereoProPainter/releases/download/weights/0a7a2514aa_inpaint_out.mp4

Right eye infilled video of moving buss:
* https://github.com/calledit/StereoProPainter/releases/download/weights/1a5fe06b00_inpaint_out.mp4


### 30 FPS video
Right eye infilled vide Moving in woods: (loots of flicker)
* https://github.com/calledit/StereoProPainter/releases/download/weights/right_eye_inpaint_out_stereo_model.mp4

Input mask visulizations:

* https://github.com/calledit/StereoProPainter/releases/download/weights/right_eye_masked_in.mp4
* https://github.com/calledit/StereoProPainter/releases/download/weights/0a7a2514aa_masked_in.mp4
* https://github.com/calledit/StereoProPainter/releases/download/weights/1a5fe06b00_masked_in.mp4

