// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Manually defined sidebar to explicitly set the order of documents and categories
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'installation', // Make sure 'installation' is the correct ID
      label: 'Installation',
    },
    {
      type: 'doc',
      id: 'overview', // Make sure 'overview' is the correct ID and you have overview.md in your docs
      label: 'Overview',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'intro', // Make sure 'intro' is the correct ID
        {
          type: 'category',
          label: 'Tutorial - Basics',
          items: [
            'tutorial-basics/create-a-document', 
            'tutorial-basics/congratulations',
            'tutorial-basics/create-a-blog-post',
            'tutorial-basics/create-a-page',
            'tutorial-basics/deploy-your-site',
            'tutorial-basics/markdown-features',
          ],
        },
        {
          type: 'category',
          label: 'Tutorial - Extras',
          items: [
            'tutorial-extras/manage-docs-versions',
            'tutorial-extras/translate-your-site',
          ],
        },
      ],
    },
    // Add additional categories or documents as needed
  ],
};

export default sidebars;
