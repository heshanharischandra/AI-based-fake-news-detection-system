DEVELOPMENT LOG
Component 4: Context-Aware Image Verification & Cross-Modal Evidence Verification
Project: Machine Learning and AI-Powered Fake News Detection System

=================================================================
DATE: 07 OCTOBER 2026
=================================================================

TASK
Initial image–text consistency research and CLIP experimentation.

COMPLETED
- Explored the NewsCLIPpings dataset for image–text verification.
- Studied matching and mismatching news image–caption pairs.
- Started experimenting with pretrained CLIP for image–text similarity.
- Inspected image annotations and associated caption information.
- Prepared a small sample of 40 image–caption examples for initial analysis.
- Organized preliminary experiment outputs into CSV files.

PROBLEMS
- CLIP produces similarity values rather than verified true/fake decisions.
- Dataset labels need careful interpretation.
- Matching image content does not prove that a news claim is factually true.

SOLUTIONS / NEXT ACTIONS
- Use labelled image–caption pairs to evaluate CLIP similarity.
- Compare matching and mismatching examples.
- Continue developing the cross-modal verification pipeline.

EVIDENCE
- NewsCLIPpings dataset files.
- Initial CLIP experiment scripts.
- CSV files containing similarity results.
- VS Code development screenshots.

=================================================================
DATE: 09 OCTOBER 2026
=================================================================

TASK
News-image dataset preparation and cross-modal analysis development.

COMPLETED
- Continued exploring the NewsCLIPpings dataset.
- Reviewed image IDs, associated captions and mismatch annotations.
- Developed initial scripts for dataset inspection.
- Organized image–caption information for cross-modal experiments.
- Generated and inspected CLIP similarity results.
- Worked on identifying possible image–headline inconsistencies.
- Prepared the initial dataset structure for further experiments.

PROBLEMS
- News image–caption datasets do not directly provide all event, time and location information.
- Raw similarity scores are insufficient to determine misinformation.
- Appropriate classification thresholds had not been established.

SOLUTIONS / NEXT ACTIONS
- Preserve original dataset labels during preprocessing.
- Separate image–text consistency from factual context verification.
- Prepare labelled examples for future classifier evaluation.
- Continue implementing additional image-verification modules.

EVIDENCE
- Dataset inspection scripts.
- NewsCLIPpings annotations.
- Generated similarity CSV files.
- Preliminary analysis outputs.

=================================================================
DATE: 12 OCTOBER 2026
=================================================================

TASK
Develop the initial AI-powered image-verification pipeline.

COMPLETED
- Organized the Python project into separate verification services.
- Integrated Tesseract OCR for Sinhala and English text extraction.
- Implemented language identification and headline-processing logic.
- Integrated a pretrained BLIP model for image description.
- Integrated a pretrained CLIP model for image–text similarity.
- Integrated a pretrained AI-generated-image classifier.
- Implemented preliminary local image-reuse detection.
- Developed reference metadata handling for context verification.
- Implemented preliminary factor-based verification scoring.
- Added rule-based explanations of available verification results.
- Developed local evidence retrieval using reference metadata and TF-IDF.

PROBLEMS
- Sinhala OCR produced inaccurate results on some news images.
- Some screenshots contained multiple photographs, logos and overlaid text.
- BLIP could generate inaccurate image descriptions.
- CLIP similarity could not establish factual truth.
- Reference evidence was insufficient for complete context verification.

SOLUTIONS / NEXT ACTIONS
- Improve image preprocessing and headline extraction.
- Separate text overlays from visual image regions.
- Avoid treating AI-generated descriptions as verified facts.
- Withhold final verification scores when evidence is insufficient.
- Continue end-to-end pipeline testing.

EVIDENCE
- Python service files.
- OCR extraction outputs.
- BLIP visual descriptions.
- CLIP similarity outputs.
- Context-verification results.
- Initial verification pipeline terminal screenshots.

=================================================================
DATE: 16 OCTOBER 2026
=================================================================

TASK
Test the complete image-verification pipeline and identify processing errors.

COMPLETED
- Tested the pipeline using news images and screenshots.
- Ran OCR, headline extraction, BLIP, CLIP, authenticity detection, image reuse and context verification.
- Tested a Sinhala news thumbnail containing multiple visual regions.
- Identified problems caused by processing an entire news screenshot as one photograph.
- Observed inaccurate Sinhala headline extraction and translation.
- Identified visual-description errors, including unsupported location predictions.
- Confirmed that the system could return an insufficient-evidence result without generating an unsupported final verification score.
- Investigated missing reference evidence and incomplete contextual checks.

PROBLEMS
- Sinhala news text was incorrectly recognized.
- OCR output mixed headline text with logos and other text.
- BLIP occasionally produced descriptions with unsupported information.
- The authenticity classifier could misinterpret composite news screenshots.
- The reference collection contained too few images for reliable reuse checking.
- Event, time, location and claim verification remained incomplete.

SOLUTIONS / NEXT ACTIONS
- Introduce a separate news-image region analysis stage.
- Allow headline review before downstream analysis.
- Improve handling of news screenshots and collages.
- Distinguish model indicators from verified facts.
- Expand the reference-image and evidence datasets.

EVIDENCE
- Sinhala news-thumbnail test results.
- OCR and translation outputs.
- BLIP and CLIP prediction logs.
- Image-authenticity outputs.
- Context verification and scoring logs.

=================================================================
DATE: 17 OCTOBER 2026
=================================================================

TASK
Improve news-image preprocessing and headline review.

COMPLETED
- Worked on separating news-image text regions from photograph regions.
- Added the news-image region analysis stage to the pipeline.
- Added a headline-review mechanism for OCR-extracted claims.
- Integrated updated processing logic into run_demo.py.
- Added support for inspecting selected news-image regions.
- Installed the updated pipeline files.
- Created backups of the previous project files.
- Confirmed that the updated Python files passed syntax checks.
- Verified that the new pipeline stages were included in the main script.

PROBLEMS
- Sinhala headlines with decorative fonts remained difficult to recognize.
- Incorrect image-region selection could affect visual analysis.
- Successful syntax checks did not establish improved OCR accuracy.
- Additional real-image testing was still required.

SOLUTIONS / NEXT ACTIONS
- Test the region analysis stage using different news layouts.
- Review OCR-generated Sinhala headlines manually when necessary.
- Improve text cropping and preprocessing.
- Compare OCR results before and after preprocessing.

EVIDENCE
- Updated run_demo.py.
- News-image analysis service.
- Installation and backup output.
- Python syntax-check results.
- Integration-marker verification.

=================================================================
DATE: 18 OCTOBER 2026
=================================================================

TASK
Test image authenticity, image–text consistency and duplicate-image detection.

COMPLETED
- Tested the system using FlyingCar.png.
- Successfully extracted the English headline: "Musk unveils flying Tesla concept".
- Generated a BLIP visual description of the futuristic car.
- Calculated a CLIP image–text similarity score of 0.3292.
- Tested the pretrained AI-image classifier.
- Recorded an AI-generated class score of 93.50% for the selected test image.
- Added or tested decision thresholds and multi-view authenticity indicators.
- Tested expanded image comparison methods, including multiple hashing methods and SIFT.
- Confirmed that no close match was found among the two available local reference images.
- Verified that the final numerical verification score remained unavailable without sufficient context evidence.

PROBLEMS
- The authenticity classifier was not calibrated.
- A high AI-generated class score could not be treated as proof of misinformation.
- Only two local reference images were available.
- Original source, event date, location and claim evidence were unavailable.
- Final verification scoring could not be completed.

SOLUTIONS / NEXT ACTIONS
- Evaluate the authenticity classifier on known real and AI-generated photographs.
- Expand the reference-image collection.
- Collect verifiable article and image sources.
- Avoid presenting model classification scores as verified truth probabilities.
- Continue integrating stronger contextual evidence.

EVIDENCE
- FlyingCar.png test output.
- OCR result.
- BLIP visual-description output.
- CLIP similarity score: 0.3292.
- AI-generation model output: 93.50%.
- Image-reuse comparison output.
- Insufficient-evidence scoring result.

=================================================================
DATE: 19 OCTOBER 2026
=================================================================

TASK
Develop and test the web-based image-verification interface.

COMPLETED
- Demonstrated the image-verification system through a local browser interface.
- Implemented news-image upload and image-preview functionality.
- Displayed extracted headline information.
- Displayed visual image descriptions and image–text similarity results.
- Displayed authenticity-model indicators.
- Displayed context-verification and evidence-coverage information.
- Tested the system using an English news image, Court.png.
- Successfully extracted the headline "Court Defers Namal Rajapaksa Bail Decision".
- Recorded a reported OCR confidence of 96% for the example.
- Calculated a CLIP similarity score of 0.3337.
- Recorded an AI-generation class score of 72.78%.
- Reviewed the interface and identified the need to make unverified results clearer.

PROBLEMS
- Sinhala headline extraction remained unreliable.
- BLIP generated a partially inaccurate image description.
- Context fields remained unassessed.
- Evidence coverage remained at 0%.
- The final verification score was pending.
- English OCR success on one example did not establish general accuracy.

SOLUTIONS / NEXT ACTIONS
- Continue improving Sinhala text extraction.
- Prepare a stable English example for PP1.
- Prepare a Sinhala example demonstrating the current limitations.
- Make pending and unverified states clear in the interface.
- Evaluate the system using multiple labelled examples.

EVIDENCE
- Web interface screenshots.
- Court.png upload and preview.
- OCR confidence output.
- CLIP similarity: 0.3337.
- AI-generation class score: 72.78%.
- Context-verification result.
- Verification-score pending status.

=================================================================
DATE: 20 OCTOBER 2026
=================================================================

TASK
Integrate the VERITE research dataset and begin database configuration.

COMPLETED — DATASET INTEGRATION
- Reviewed the dataset requirements for OCR, cross-modal verification, authenticity analysis, image reuse and context checking.
- Organized the dataset folder structure inside the VS Code project.
- Downloaded and extracted the official VERITE dataset builder.
- Identified a source-data count discrepancy between 1,000 expected records and 1,001 available records.
- Used a corrected dataset builder to create a documented 1,000-record subset.
- Generated the dataset inside dataset/news_image_context_verification_dataset/.
- Generated master_news_images.csv with 1,000 image–caption records.
- Generated cross_modal/all_pairs.csv with 1,000 records.
- Generated separate training, validation and testing CSV files.
- Recorded 799 training, 102 validation and 99 testing records.
- Created corresponding OCR, authenticity, visual and context CSV files.
- Generated an image-reuse file containing 338 supported comparison records.
- Inspected dataset files in VS Code and investigated CSV display issues.

COMPLETED — MONGODB CONFIGURATION
- Opened the MongoDB Atlas dashboard.
- Identified the available cloud cluster.
- Selected Python/PyMongo as the connection method.
- Obtained the connection-string format for backend configuration.
- Began preparing environment-variable configuration for the database connection.

PROBLEMS
- The dataset builder initially stopped because the upstream file contained 1,001 records.
- Some original online image URLs might no longer be accessible.
- VERITE does not contain Sinhala news examples.
- The benchmark does not provide verified AI-generation labels or complete OCR transcriptions.
- VS Code's CSV editor displayed a license-related warning.
- MongoDB connection and result storage had not yet been confirmed through a successful Python test.

SOLUTIONS / NEXT ACTIONS
- Preserve the documented record-selection details.
- Validate the generated CSV structure and record counts.
- Check whether original image URLs remain accessible.
- Keep missing authenticity and context fields unverified.
- Collect separate Sinhala and Sri Lankan news-image examples.
- Complete MongoDB connection testing.
- Integrate database storage with verification outputs.
- Prepare the dataset and prototype demonstrations for PP1.

EVIDENCE
- VERITE dataset generation output.
- master_news_images.csv.
- cross_modal/all_pairs.csv.
- Training, validation and testing CSV files.
- Dataset folder screenshots.
- MongoDB Atlas dashboard screenshots.
- Python connection configuration details.

=================================================================
DATE: 21 OCTOBER 2026
=================================================================

TASK
Final PP1 system testing and presentation preparation.

STATUS
Planned — not yet confirmed completed.

PLANNED WORK
- Test the English image-verification demonstration.
- Test Sinhala OCR using a clear headline example.
- Record OCR errors and available correction methods.
- Validate the dataset CSV counts and selected image URLs.
- Capture system screenshots and model outputs.
- Prepare the system architecture explanation.
- Prepare a summary of implemented AI models and algorithms.
- Document incomplete components and research limitations.
- Practice the PP1 demonstration and technical explanations.

EXPECTED EVIDENCE
- Final demonstration screenshots.
- Sinhala and English test results.
- Dataset validation report.
- Presentation slides.
- PP1 system demonstration.

=================================================================
DATE: 22 OCTOBER 2026
=================================================================

TASK
Progress Presentation 1 (PP1).

STATUS
Scheduled.

DEMONSTRATION SCOPE
- News-image upload and preprocessing.
- Sinhala and English OCR functionality.
- BLIP visual description.
- CLIP image–text similarity.
- Preliminary AI-generated-image detection.
- Local image-reuse checking.
- Preliminary context verification.
- Verification scoring and Explainable AI.
- Dataset organization and labelled benchmark examples.
- Discussion of limitations and future improvements.

=================================================================
OVERALL DEVELOPMENT SUMMARY
=================================================================

IMPLEMENTED PROTOTYPE FEATURES
- News-image upload and processing.
- English OCR and preliminary Sinhala OCR.
- BLIP visual-description integration.
- CLIP image–text similarity integration.
- Pretrained AI-generation classifier integration.
- Local image-reuse detection.
- Preliminary context verification.
- Rule-based scoring and explanations.
- Local evidence retrieval.
- Web-based result visualization.
- VERITE benchmark CSV integration.

CURRENT LIMITATIONS
- Sinhala headline extraction accuracy needs improvement.
- BLIP descriptions and authenticity scores require formal evaluation.
- Image-text matching thresholds are not calibrated.
- Context verification lacks sufficient verified reference evidence.
- Original image URLs require availability checks.
- Numerical verification scores are often withheld because evidence is insufficient.
- Research-specific model training has not been confirmed.
- MongoDB result storage integration is not yet confirmed working.

NEXT DEVELOPMENT PHASE
- Sinhala OCR optimization.
- Verified Sri Lankan and international news-image collection.
- Model performance evaluation.
- Improved image authenticity and manipulation analysis.
- Event, time, location and claim verification.
- Verification-score calibration.
- Database and final application integration.
- Final experimental evaluation and documentation.

=================================================================
RECOMMENDED MICROSOFT TEAMS PLANNER CARD TITLES
=================================================================
