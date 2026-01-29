# rectangle and point coordinates for correlation analysis


experiment_db = {
    # Training data
    "simdata_ref_a_xclower=0.15": {
        "rectangles": [(25, 120, 30, 135), ],
        "points":[(39, 130), (39, 165), (39, 200), (39, 245)]
    },
    
      "simdata_ref_a_xclower=0.20": {
        "rectangles": [(35, 120, 30, 135), ],
        "points": [(49, 130), (49, 165), (49, 200), (49, 245)]
    },
    
      "simdata_ref_a_xclower=0.22": {
        "rectangles": [(42, 120, 30, 135), ],
        "points": [(53, 130), (53, 165), (53, 200), (53, 245)]
    },
    
      "simdata_ref_a_xclower=0.25": {
        "rectangles": [(49, 120, 30, 135), ],
        "points": [(61, 130), (61, 165), (61, 200), (61, 245)]
    },
    
      "simdata_ref_a_xclower=0.30": {
        "rectangles": [(62, 120, 30, 135), ],
        "points": [(74, 130), (74, 165), (74, 200), (74, 245)]
    },
    
      "simdata_ref_a_xclower=0.35": {
        "rectangles": [(75, 120, 30, 135), ],
        "points": [(88, 130), (88, 165), (88, 200), (88, 245)]
    },
    
      "simdata_ref_a_xclower=0.40": {
        "rectangles": [(85, 120, 30, 135), ],
        "points": [(99, 130), (99, 165), (99, 200), (99, 245)]
    },
    
      "simdata_ref_a_xclower=0.45": {
        "rectangles": [(90, 120, 30, 135), ],
        "points": [(109, 130), (109, 165), (109, 200), (109, 245)]
    },
    
      "simdata_ref_a_xclower=0.48": {
        "rectangles": [(108, 120, 30, 135), ],
        "points": [(117, 130), (107, 165), (107, 200), (97, 245)]
    },
      
        "simdata_ref_a_xclower=0.50": {
        "rectangles": [(82, 120, 30, 135),(144, 120, 30, 135),  ],
        "points": [(128, 130), (95, 162), (157, 162), (40, 245), (215,245) ]
    },
      
        "simdata_ref_a_xclower=0.52": {
        "rectangles": [(122, 120, 30, 135)],
        "points": [(128, 130), (145, 165), (145, 200), (153, 245)]
    },
      
        "simdata_ref_a_xclower=0.55": {
        "rectangles": [(140, 120, 30, 135)],
        "points": [(140, 130), (140, 165), (140, 200), (140, 245)]
    },
      
        "simdata_ref_a_xclower=0.60": {
        "rectangles": [(140, 120, 30, 135)],
        "points": [(151, 130), (151, 165), (151, 200), (151, 245)]
    },
      
        "simdata_ref_a_xclower=0.65": {
        "rectangles": [(153, 120, 30, 135)],
        "points": [(162, 130), (162, 165), (162, 200), (162, 245)]
    },
      
        "simdata_ref_a_xclower=0.70": {
        "rectangles": [(163, 120, 30, 135)],
        "points": [(173, 130), (173, 165), (173, 200), (173, 245)]
    },
      
        "simdata_ref_a_xclower=0.75": {
        "rectangles": [(176, 120, 30, 135)],
        "points": [(190, 130), (190, 165), (190, 200), (190, 245)]  },
      
        "simdata_ref_a_xclower=0.78": {
        "rectangles": [(182, 120, 30, 135)],
        "points": [(198, 130), (198, 165), (198, 200), (198, 245)]
    },
      
        "simdata_ref_a_xclower=0.80": {
        "rectangles": [(189, 120, 30, 135)],
        "points": [(203, 130), (203, 165), (203, 200), (203, 245)]
    },
      
        "simdata_ref_a_xclower=0.85": {
        "rectangles": [(201, 120, 30, 135)],
        "points": [(211, 130), (211, 165), (211, 200), (211, 245)]
    },
        
 

}


def get_experiment_info(experiment_name):
    return experiment_db.get(experiment_name, None)  # fallback to full DB