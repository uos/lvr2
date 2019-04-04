//
// Created by patrick on 4/4/19.
//


namespace lvr2{

    TumbleTree::node* TumbleTree::makeEmpty(node* t)
    {
        if(t == NULL)
            return NULL;
        {
            makeEmpty(t->left);
            makeEmpty(t->right);
            delete t;
        }
        return NULL;
    }

    TumbleTree::node* TumbleTree::insert(float sc, int vH, node* t)
    {
        if(t == NULL)
        {
            t = new TumbleTree::node;
            t->signal_counter = sc;
            t->vH = vH;
            t->left = t->right = NULL;
        }
        else if(sc < t->signal_counter)
            t->left = insert(sc, vH, t->left);
        else if(sc > t->signal_counter)
            t->right = insert(sc, vH, t->right);
        return t;
    }

    TumbleTree::node* TumbleTree::findMin(node* t)
    {
        if(t == NULL)
            return NULL;
        else if(t->left == NULL)
            return t;
        else
            return findMin(t->left);
    }

    TumbleTree::node* TumbleTree::findMax(node* t)
    {
        if(t == NULL)
            return NULL;
        else if(t->right == NULL)
            return t;
        else
            return findMax(t->right);
    }

    TumbleTree::node* TumbleTree::remove(float sc, node* t)
    {
        node* temp;
        if(t == NULL)
            return NULL;
        else if(sc < t->signal_counter)
            t->left = remove(sc, t->left);
        else if(sc > t->signal_counter)
            t->right = remove(sc, t->right);
        else if(t->left && t->right)
        {
            temp = findMin(t->right);
            t->signal_counter = temp->signal_counter;
            t->right = remove(t->signal_counter, t->right);
        }
        else
        {
            temp = t;
            if(t->left == NULL)
                t = t->right;
            else if(t->right == NULL)
                t = t->left;
            delete temp;
        }

        return t;
    }

    void TumbleTree::inorder(node* t)
    {
        if(t == NULL)
            return;
        inorder(t->left);
        cout << t->signal_counter << " ";
        inorder(t->right);
    }

    TumbleTree::node* TumbleTree::find(node* t, float sc) {
        if (t == NULL)
            return NULL;
        else if (sc < t->signal_counter)
            return find(t->left, sc);
        else if (sc > t->signal_counter)
            return find(t->right, sc);
        else
            return t;
    }

    TumbleTree::TumbleTree()
    {
        root = NULL;
    }

    TumbleTree::~TumbleTree()
    {
        root = makeEmpty(root);
    }

    void TumbleTree::insert(int x, VertexHandle vH)
    {
        root = insert(x, vH.idx(), root);
    }

    void TumbleTree::remove(int x)
    {
        root = remove(x, root);
    }

    void TumbleTree::display()
    {
        inorder(root);
        cout << endl;
    }

    VertexHandle TumbleTree::min()
    {
        return VertexHandle(this->findMin(root)->vH);
    }

    VertexHandle TumbleTree::max()
    {
        return VertexHandle(this->findMax(root)->vH);
    }

    void TumbleTree::update(float alpha) {

    }

} // namespace lvr2