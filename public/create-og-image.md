# Creating the Open Graph Image for LinkedIn

## Steps to create og-image.png:

1. **Option A: Use the HTML template**
   - Open `public/og-image-template.html` in your browser
   - Take a screenshot at exactly 1200x630 pixels
   - Save as `og-image.png` in the public folder

2. **Option B: Use an online tool**
   - Go to https://www.bannerbear.com/tools/linkedin-post-preview/
   - Or use https://www.canva.com/ (search for "LinkedIn Article Cover" template)
   - Create a 1200x630px image with:
     - Your EvolvIQ logo
     - Title: "RAG Implementation Guide"
     - Subtitle: "Build Production-Ready AI Knowledge Systems"
     - Your brand colors (#A44A3F, #A59E8C, #D7CEB2, #F5F2EA)

3. **Option C: Use a design tool**
   - Figma, Sketch, or Adobe XD
   - Create a 1200x630px artboard
   - Use the design from og-image-template.html as reference

## After creating the image:

1. Save it as `og-image.png` in your `public` folder
2. Update the Vercel URL in `public/index.html` (replace `rag-implementation-guide.vercel.app` with your actual URL)
3. Deploy to Vercel
4. Test your link with LinkedIn's Post Inspector: https://www.linkedin.com/post-inspector/

## LinkedIn Requirements:
- Minimum size: 1200x627 pixels
- Maximum file size: 5MB
- Format: PNG or JPG
- Must be served over HTTPS

## Troubleshooting:
- Clear LinkedIn's cache using their Post Inspector
- Make sure the image URL is absolute (https://...)
- Ensure no authentication is required to access the image
- Wait a few minutes after deployment for CDN propagation